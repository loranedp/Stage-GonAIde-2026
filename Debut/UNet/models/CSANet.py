"""CSA-Net pour la segmentation de sequences de coupes 2.5D.

L'article original predit uniquement la coupe centrale d'un triplet
``(precedente, centrale, suivante)``. Cette adaptation applique le meme calcul a
chaque coupe d'une sequence et conserve donc l'interface historique du projet :

    (B, T, C, H, W) -> (B, T, num_classes, H, W)

Aux extremites, la coupe centrale remplace le voisin manquant.
"""

import copy

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        encoder = smp.encoders.get_encoder(
            "resnet50",
            in_channels=in_channels,
            depth=4,
            weights=encoder_weights,
        )
        # ``depth=4`` s'arrete a layer3, mais SMP conserve layer4 comme
        # sous-module enregistre. Il ne participerait jamais au forward tout en
        # gonflant inutilement le state_dict et l'optimiseur.
        encoder.layer4 = nn.Identity()
        self.center_encoder = encoder
        self.previous_encoder = copy.deepcopy(encoder)
        self.next_encoder = copy.deepcopy(encoder)

        feature_channels = encoder.out_channels[-1]  # 1024 a H/16
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

        center_features = self.center_encoder(center_images)
        previous = self.previous_encoder(previous_images)[-1]
        following = self.next_encoder(next_images)[-1]
        return center_features, previous, following

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("CSANet attend une entree (B, T, C, H, W)")
        batch, length, _, input_height, input_width = x.shape
        if length < 1:
            raise ValueError("Une sequence doit contenir au moins une coupe")

        center_features, previous, following = self._encode_triplet(x)
        center = center_features[-1]

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

        skips = (center_features[3], center_features[2], center_features[1], None)
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
