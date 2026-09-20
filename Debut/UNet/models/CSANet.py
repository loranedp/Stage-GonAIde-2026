"""CSA-Net pour la segmentation de sequences de coupes 2.5D.

L'article original predit uniquement la coupe centrale d'un triplet
``(precedente, centrale, suivante)``. Cette adaptation applique le meme calcul a
chaque coupe d'une sequence et conserve donc l'interface historique du projet :

    (B, T, C, H, W) -> (B, T, num_classes, H, W)

Aux extremites, la coupe centrale remplace le voisin manquant.
"""

import copy
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _np_to_tensor(array, convolution=False):
    """Convertit les poids JAX (HWIO) au format PyTorch (OIHW)."""
    if convolution:
        array = array.transpose(3, 2, 0, 1)
    return torch.from_numpy(array)


class StdConv2d(nn.Conv2d):
    """Convolution avec standardisation des poids utilisee par ResNetV2."""

    def forward(self, x):
        variance, mean = torch.var_mean(
            self.weight, dim=(1, 2, 3), keepdim=True, unbiased=False
        )
        weight = (self.weight - mean) / torch.sqrt(variance + 1e-5)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PreActBottleneck(nn.Module):
    """Bottleneck preactivation ResNetV2 du checkpoint R50+ViT-B/16."""

    def __init__(self, in_channels, out_channels, middle_channels, stride=1):
        super().__init__()
        self.conv1 = StdConv2d(in_channels, middle_channels, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, middle_channels, eps=1e-6)
        self.conv2 = StdConv2d(
            middle_channels,
            middle_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(32, middle_channels, eps=1e-6)
        self.conv3 = StdConv2d(middle_channels, out_channels, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = StdConv2d(
                in_channels, out_channels, 1, stride=stride, bias=False
            )
            self.gn_proj = nn.GroupNorm(out_channels, out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.gn_proj(self.downsample(x))

        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        return self.relu(residual + x)

    def load_from(self, weights, block_name, unit_name):
        prefix = f"{block_name}/{unit_name}"
        with torch.no_grad():
            for name in ("conv1", "conv2", "conv3"):
                getattr(self, name).weight.copy_(
                    _np_to_tensor(weights[f"{prefix}/{name}/kernel"], True)
                )
            for name in ("gn1", "gn2", "gn3"):
                module = getattr(self, name)
                module.weight.copy_(_np_to_tensor(weights[f"{prefix}/{name}/scale"]).view(-1))
                module.bias.copy_(_np_to_tensor(weights[f"{prefix}/{name}/bias"]).view(-1))

            if self.downsample is not None:
                self.downsample.weight.copy_(
                    _np_to_tensor(weights[f"{prefix}/conv_proj/kernel"], True)
                )
                self.gn_proj.weight.copy_(
                    _np_to_tensor(weights[f"{prefix}/gn_proj/scale"]).view(-1)
                )
                self.gn_proj.bias.copy_(
                    _np_to_tensor(weights[f"{prefix}/gn_proj/bias"]).view(-1)
                )


class ResNetV2(nn.Module):
    """Partie ResNet-50 hybride officielle, jusqu'au bottleneck H/16."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.root = nn.Sequential(
            OrderedDict(
                (
                    (
                        "conv",
                        StdConv2d(
                            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, 64, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                )
            )
        )
        self.body = nn.Sequential(
            OrderedDict(
                (
                    ("block1", self._make_block(64, 256, 64, units=3, stride=1)),
                    ("block2", self._make_block(256, 512, 128, units=4, stride=2)),
                    ("block3", self._make_block(512, 1024, 256, units=9, stride=2)),
                )
            )
        )

    @staticmethod
    def _make_block(in_channels, out_channels, middle_channels, units, stride):
        layers = [
            (
                "unit1",
                PreActBottleneck(
                    in_channels, out_channels, middle_channels, stride=stride
                ),
            )
        ]
        layers.extend(
            (f"unit{index}", PreActBottleneck(out_channels, out_channels, middle_channels))
            for index in range(2, units + 1)
        )
        return nn.Sequential(OrderedDict(layers))

    @staticmethod
    def _pad_to(x, target_height, target_width):
        pad_height = target_height - x.shape[-2]
        pad_width = target_width - x.shape[-1]
        if pad_height < 0 or pad_width < 0 or pad_height > 2 or pad_width > 2:
            raise ValueError(
                f"Taille de feature inattendue {tuple(x.shape[-2:])}, "
                f"attendue {(target_height, target_width)}"
            )
        return F.pad(x, (0, pad_width, 0, pad_height))

    def forward(self, x):
        input_height, input_width = x.shape[-2:]
        root = self.root(x)
        x = F.max_pool2d(root, kernel_size=3, stride=2, padding=0)

        block1 = self.body.block1(x)
        skip_quarter = self._pad_to(
            block1, input_height // 4, input_width // 4
        )
        block2 = self.body.block2(block1)
        skip_eighth = self._pad_to(
            block2, input_height // 8, input_width // 8
        )
        bottleneck = self.body.block3(block2)
        return bottleneck, (skip_eighth, skip_quarter, root)

    def load_from(self, weights):
        with torch.no_grad():
            root_kernel = _np_to_tensor(weights["conv_root/kernel"], True)
            if root_kernel.shape[1] != self.root.conv.in_channels:
                raise ValueError(
                    "Le checkpoint R50+ViT-B_16 requiert des images a 3 canaux"
                )
            self.root.conv.weight.copy_(root_kernel)
            self.root.gn.weight.copy_(_np_to_tensor(weights["gn_root/scale"]).view(-1))
            self.root.gn.bias.copy_(_np_to_tensor(weights["gn_root/bias"]).view(-1))
            for block_name, block in self.body.named_children():
                for unit_name, unit in block.named_children():
                    unit.load_from(weights, block_name, unit_name)


class CrossSliceAttention(nn.Module):
    """Attention non-locale multi-tete de la coupe voisine vers le centre."""

    def __init__(self, channels=1024, num_heads=16):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels doit etre divisible par num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads

        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=num_heads),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.output[1].weight)
        nn.init.zeros_(self.output[1].bias)

    def _as_heads(self, tensor):
        batch, _, height, width = tensor.shape
        return tensor.reshape(
            batch, self.num_heads, self.head_channels, height * width
        )

    def forward(self, center, neighbor):
        if center.shape != neighbor.shape:
            raise ValueError("center et neighbor doivent avoir la meme forme")

        batch, _, height, width = center.shape
        # Q: pixels du voisin ; K/V: pixels de la coupe centrale.
        query = self._as_heads(self.query(neighbor)).transpose(-1, -2)
        key = self._as_heads(self.key(center)).transpose(-1, -2)
        value = self._as_heads(self.value(center)).transpose(-1, -2)

        # L'article et l'implementation officielle n'appliquent pas le facteur
        # 1/sqrt(d) du Transformer standard. SDPA permet au backend CUDA
        # d'employer son noyau memoire-efficace sans materialiser toute la
        # matrice (H*W)x(H*W).
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, scale=1.0
        )
        attended = attended.transpose(-1, -2).reshape(
            batch, self.channels, height, width
        )
        return self.output(attended)


class ConvNormReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.conv1 = ConvNormReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvNormReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat((x, skip), dim=1)
        return self.conv2(self.conv1(x))


class CSANet(nn.Module):
    """CSA-Net fidele a l'article, etendue pour predire les ``T`` coupes."""

    def __init__(
        self,
        num_classes,
        in_channels,
        encoder_weights="imagenet",
        num_heads=16,
        image_size=512,
        transformer_layers=12,
        transformer_heads=12,
        hidden_size=768,
        mlp_dim=3072,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_size % transformer_heads != 0:
            raise ValueError("hidden_size doit etre divisible par transformer_heads")

        if encoder_weights not in (None, "imagenet"):
            raise ValueError("encoder_weights doit valoir 'imagenet' ou None")
        encoder = ResNetV2(in_channels=in_channels)
        self.center_encoder = encoder
        self.previous_encoder = copy.deepcopy(encoder)
        self.next_encoder = copy.deepcopy(encoder)

        feature_channels = 1024
        if feature_channels % num_heads != 0:
            raise ValueError(
                "Le nombre de canaux du ResNet-50 doit etre divisible par num_heads"
            )

        self.previous_attention = CrossSliceAttention(feature_channels, num_heads)
        self.next_attention = CrossSliceAttention(feature_channels, num_heads)
        self.in_slice_attention = CrossSliceAttention(feature_channels, num_heads)
        self.attention_fusion = ConvNormReLU(
            3 * feature_channels, feature_channels, kernel_size=1
        )

        self.patch_projection = nn.Conv2d(feature_channels, hidden_size, kernel_size=1)
        reference_grid = max(1, image_size // 16)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, hidden_size, reference_grid, reference_grid)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, norm=nn.LayerNorm(hidden_size)
        )

        self.decoder = nn.ModuleList(
            (
                DecoderBlock(hidden_size, 256, skip_channels=512),
                DecoderBlock(256, 128, skip_channels=256),
                DecoderBlock(128, 64, skip_channels=64),
                DecoderBlock(64, 16),
            )
        )
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

        if encoder_weights == "imagenet":
            pretrained_path = Path(__file__).with_name("R50+ViT-B_16.npz")
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, checkpoint_path):
        """Charge le checkpoint officiel hybride R50+ViT-B/16."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint CSA-Net preentraine introuvable: {checkpoint_path}"
            )
        if len(self.transformer.layers) != 12:
            raise ValueError("Le checkpoint officiel requiert transformer_layers=12")
        first_layer = self.transformer.layers[0]
        if (
            first_layer.self_attn.embed_dim != 768
            or first_layer.self_attn.num_heads != 12
            or first_layer.linear1.out_features != 3072
        ):
            raise ValueError(
                "Le checkpoint officiel requiert hidden_size=768, "
                "transformer_heads=12 et mlp_dim=3072"
            )

        try:
            weights = np.load(checkpoint_path)
        except Exception as error:
            raise ValueError(
                f"Impossible de lire le checkpoint {checkpoint_path}: {error}"
            ) from error

        try:
            self.center_encoder.load_from(weights)
            encoder_state = self.center_encoder.state_dict()
            self.previous_encoder.load_state_dict(encoder_state)
            self.next_encoder.load_state_dict(encoder_state)

            with torch.no_grad():
                self.patch_projection.weight.copy_(
                    _np_to_tensor(weights["embedding/kernel"], convolution=True)
                )
                self.patch_projection.bias.copy_(
                    _np_to_tensor(weights["embedding/bias"])
                )

                position = _np_to_tensor(
                    weights["Transformer/posembed_input/pos_embedding"]
                )
                # Le premier token est le token de classification du ViT.
                position = position[:, 1:]
                source_grid = math.isqrt(position.shape[1])
                if source_grid * source_grid != position.shape[1]:
                    raise ValueError("Embedding de position non carre dans le checkpoint")
                position = position.reshape(1, source_grid, source_grid, -1)
                position = position.permute(0, 3, 1, 2)
                position = F.interpolate(
                    position,
                    size=self.position_embedding.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                self.position_embedding.copy_(position)

                for index, layer in enumerate(self.transformer.layers):
                    prefix = f"Transformer/encoderblock_{index}"
                    qkv_weights = []
                    qkv_biases = []
                    for name in ("query", "key", "value"):
                        kernel = _np_to_tensor(
                            weights[
                                f"{prefix}/MultiHeadDotProductAttention_1/{name}/kernel"
                            ]
                        ).reshape(768, 768)
                        qkv_weights.append(kernel.t())
                        qkv_biases.append(
                            _np_to_tensor(
                                weights[
                                    f"{prefix}/MultiHeadDotProductAttention_1/{name}/bias"
                                ]
                            ).reshape(-1)
                        )
                    layer.self_attn.in_proj_weight.copy_(torch.cat(qkv_weights, dim=0))
                    layer.self_attn.in_proj_bias.copy_(torch.cat(qkv_biases, dim=0))

                    output_kernel = _np_to_tensor(
                        weights[
                            f"{prefix}/MultiHeadDotProductAttention_1/out/kernel"
                        ]
                    ).reshape(768, 768)
                    layer.self_attn.out_proj.weight.copy_(output_kernel.t())
                    layer.self_attn.out_proj.bias.copy_(
                        _np_to_tensor(
                            weights[
                                f"{prefix}/MultiHeadDotProductAttention_1/out/bias"
                            ]
                        )
                    )

                    layer.linear1.weight.copy_(
                        _np_to_tensor(
                            weights[f"{prefix}/MlpBlock_3/Dense_0/kernel"]
                        ).t()
                    )
                    layer.linear1.bias.copy_(
                        _np_to_tensor(weights[f"{prefix}/MlpBlock_3/Dense_0/bias"])
                    )
                    layer.linear2.weight.copy_(
                        _np_to_tensor(
                            weights[f"{prefix}/MlpBlock_3/Dense_1/kernel"]
                        ).t()
                    )
                    layer.linear2.bias.copy_(
                        _np_to_tensor(weights[f"{prefix}/MlpBlock_3/Dense_1/bias"])
                    )
                    layer.norm1.weight.copy_(
                        _np_to_tensor(weights[f"{prefix}/LayerNorm_0/scale"])
                    )
                    layer.norm1.bias.copy_(
                        _np_to_tensor(weights[f"{prefix}/LayerNorm_0/bias"])
                    )
                    layer.norm2.weight.copy_(
                        _np_to_tensor(weights[f"{prefix}/LayerNorm_2/scale"])
                    )
                    layer.norm2.bias.copy_(
                        _np_to_tensor(weights[f"{prefix}/LayerNorm_2/bias"])
                    )

                self.transformer.norm.weight.copy_(
                    _np_to_tensor(weights["Transformer/encoder_norm/scale"])
                )
                self.transformer.norm.bias.copy_(
                    _np_to_tensor(weights["Transformer/encoder_norm/bias"])
                )
        except (KeyError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"Checkpoint incompatible {checkpoint_path}: {error}"
            ) from error
        finally:
            weights.close()

    @staticmethod
    def _neighbor_indices(length, device):
        """Indices precedents/suivants, avec la coupe centrale aux bords."""
        indices = torch.arange(length, device=device)
        return (indices - 1).clamp_min(0), (indices + 1).clamp_max(length - 1)

    def _encode_triplet(self, x):
        batch, length, channels, height, width = x.shape
        previous_idx, next_idx = self._neighbor_indices(length, x.device)

        center_images = x.reshape(batch * length, channels, height, width)
        previous_images = x[:, previous_idx].reshape(
            batch * length, channels, height, width
        )
        next_images = x[:, next_idx].reshape(batch * length, channels, height, width)

        center, center_skips = self.center_encoder(center_images)
        previous, _ = self.previous_encoder(previous_images)
        following, _ = self.next_encoder(next_images)
        return center, center_skips, previous, following

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("CSANet attend une entree (B, T, C, H, W)")
        batch, length, _, input_height, input_width = x.shape
        if length < 1:
            raise ValueError("Une sequence doit contenir au moins une coupe")

        center, center_skips, previous, following = self._encode_triplet(x)

        attended = torch.cat(
            (
                self.previous_attention(center, previous),
                self.in_slice_attention(center, center),
                self.next_attention(center, following),
            ),
            dim=1,
        )
        hidden = self.attention_fusion(attended)
        hidden = self.patch_projection(hidden)

        position = F.interpolate(
            self.position_embedding,
            size=hidden.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = hidden + position
        grid_height, grid_width = hidden.shape[-2:]
        tokens = hidden.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        hidden = tokens.transpose(1, 2).reshape(
            batch * length, -1, grid_height, grid_width
        )

        skips = (*center_skips, None)
        for block, skip in zip(self.decoder, skips):
            hidden = block(hidden, skip)

        logits = self.segmentation_head(hidden)
        if logits.shape[-2:] != (input_height, input_width):
            logits = F.interpolate(
                logits,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            )
        return logits.reshape(batch, length, -1, input_height, input_width)
