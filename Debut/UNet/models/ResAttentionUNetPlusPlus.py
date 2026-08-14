# Définition de l'architecture ResAttentionUNetPlusPlus

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# Evalue l'importance des canaux pour améliorer la segmentation
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Résumé spatial de chaque canal en un seul pixel
        self.mlp = nn.Sequential( # Réseau de neurones pour l'attention des canaux
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid() # Score d'importance des canaux entre 0 et 1

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

# Evalue ou se situe l'attention spatiale pour améliorer la segmentation
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid() # Score d'attention spatiale entre 0 et 1

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

# Bloc de fusion de l'attention des canaux et de l'attention spatiale avec un skip connection
class ResAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return x + out # Skip connection

# Modèle final : U-Net++ avec attention résiduelle
class ResAttentionUNetPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels, encoder_weights="imagenet"):
        super().__init__()
        self.base_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        decoder_out_channels = self.base_model.segmentation_head[0].in_channels
        self.attention_block = ResAttentionBlock(in_channels=decoder_out_channels)

    def forward(self, x):
        features = self.base_model.encoder(x) # Passe dans l'encodeur pour extraire les caractéristiques
        decoder_output = self.base_model.decoder(features) # Passe dans le décodeur pour reconstruire la segmentation
        refined_output = self.attention_block(decoder_output) # Applique l'attention résiduelle pour améliorer la segmentation
        logits = self.base_model.segmentation_head(refined_output) # Tête de segmentation finale pour produire les logits
        return logits
