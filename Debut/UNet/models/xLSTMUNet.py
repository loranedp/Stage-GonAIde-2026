# xLSTM-UNet
# Variante qui remplace l'attention inter-coupes par un bloc **xLSTM** (Extended LSTM, Beck et al. 2024) traitant la séquence de coupes de façon récurrente plutôt que par attention.

import segmentation_models_pytorch as smp
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig

class _PerSliceUNetBase(nn.Module):
    def __init__(self, num_classes, in_channels, encoder_weights="imagenet"):
        super().__init__()
        self.base_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        self.mix = None  # module de mélange inter-coupes, défini par les sous-classes

    def forward(self, x):
        # x: (B, T, C, H, W) - séquence de coupes échographiques ordonnées par position
        B, T, C, H, W = x.shape
        features = self.base_model.encoder(x.view(B * T, C, H, W)) # Encodage indépendant de chaque coupe

        # Mélange inter-coupes sur le bottleneck (feature la plus profonde) uniquement
        _, Cb, Hb, Wb = features[-1].shape
        bottleneck = features[-1].view(B, T, Cb, Hb, Wb)
        bottleneck = self.mix(bottleneck)
        features[-1] = bottleneck.reshape(B * T, Cb, Hb, Wb)

        decoder_output = self.base_model.decoder(features) # Décodage indépendant de chaque coupe
        logits = self.base_model.segmentation_head(decoder_output)
        return logits.view(B, T, -1, H, W)


# Attention multi-tête entre coupes : chaque position spatiale du bottleneck attend
# aux autres coupes de la séquence (indépendamment des autres positions spatiales)
class CrossSliceAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C) # Une séquence de longueur T par position spatiale
        attn_out, _ = self.attn(seq, seq, seq)
        out = self.norm(seq + attn_out) # Résidu + normalisation
        return out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)


# Mélange inter-coupes via un bloc xLSTM (mLSTM) traitant la séquence le long de T,
# indépendamment pour chaque position spatiale du bottleneck.
# max_seq_len doit être >= au nombre de coupes T réellement utilisé par séquence.
class xLSTMSliceMixer(nn.Module):
    def __init__(self, channels, num_blocks=2, max_seq_len=16):
        super().__init__()
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(embedding_dim=channels, context_length=max_seq_len)),
            num_blocks=num_blocks,
            embedding_dim=channels,
            context_length=max_seq_len,
        )
        self.xlstm = xLSTMBlockStack(cfg)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C) # Une séquence de longueur T par position spatiale
        out = self.xlstm(seq)
        return out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)

# Modèle final : xLSTM-UNet
class xLSTMUNet(_PerSliceUNetBase):
    def __init__(self, num_classes, in_channels, encoder_weights="imagenet", num_blocks=2, max_seq_len=16):
        super().__init__(num_classes, in_channels, encoder_weights)
        bottleneck_channels = self.base_model.encoder.out_channels[-1]
        self.mix = xLSTMSliceMixer(bottleneck_channels, num_blocks=num_blocks, max_seq_len=max_seq_len)