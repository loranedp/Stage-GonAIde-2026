## CSA-Net et xLSTM-UNet — dépendance entre coupes échographiques

#Les architectures précédentes traitent chaque image indépendamment. Or les images d'un même poisson forment une **séquence de coupes échographiques ordonnées** (voir `position` / `position_ratio` dans les métadonnées), qui s'apparente à un volume 3D pseudo-continu.

#Les deux modèles ci-dessous partagent le même principe :
#- Chaque coupe de la séquence est encodée indépendamment par l'encodeur ResNet34 (poids partagés entre les coupes).
#- Les features du **bottleneck** (la représentation la plus profonde) sont mélangées le long de la dimension des coupes `T`, pour propager le contexte entre coupes voisines.
#- Chaque coupe est ensuite décodée indépendamment par le décodeur U-Net habituel (skip connections non mélangées).

import segmentation_models_pytorch as smp
import torch.nn as nn


# Classe de base commune : encode chaque coupe indépendamment (ResNet34 partagé),
# mélange les features du bottleneck le long de la dimension des coupes T,
# puis décode chaque coupe indépendamment.
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

# Modèle final : CSA-Net (Cross-Slice Attention Net)
class CSANet(_PerSliceUNetBase):
    def __init__(self, num_classes, in_channels, encoder_weights="imagenet", num_heads=4):
        super().__init__(num_classes, in_channels, encoder_weights)
        bottleneck_channels = self.base_model.encoder.out_channels[-1]
        self.mix = CrossSliceAttention(bottleneck_channels, num_heads=num_heads)