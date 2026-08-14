### Attention Gated Multi ResU-Net

#- Blocs *MultiRes* (résiduels multi-résolution) dans l'encodeur et le décodeur
#- *Attention gates* sur les connexions de saut (skip connections)
#- 5 blocs d'encodeur / décodeur, filtres 32→64→128→256→512
#- Entrée 256×256 -> 512x512
#- Transformation segmentation **binaire** en **multi-classes** 

from ultralytics import nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """BatchNorm -> Conv -> ReLU (ordre BN puis conv, comme décrit dans l\'article)."""

    def __init__(self, in_ch, out_ch, k=3, act=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(self.bn(x))
        return self.act(x) if self.act is not None else x

# MultiResBlock (pour regarder les images à différentes résolutions)
class MultiResBlock(nn.Module):
    """Bloc MultiRes : 3 convs 3x3 chaînées concaténées + raccourci résiduel 1x1."""

    def __init__(self, in_ch, U):
        super().__init__()
        f1, f2 = U // 6, U // 3
        f3 = U - f1 - f2  # somme = U -> canaux de sortie exactement U
        self.c1 = ConvBNAct(in_ch, f1) # 1ère convolution 3x3 (détails fins)
        self.c2 = ConvBNAct(f1, f2) # 2ème convolution 5x5 (détails moyens)
        self.c3 = ConvBNAct(f2, f3) # 3ème convolution 7x7 (formes globales)
        self.shortcut = ConvBNAct(in_ch, U, k=1, act=False)  # H(x)
        self.bn = nn.BatchNorm2d(U)
        self.out_ch = U

    def forward(self, x):
        s = self.shortcut(x)
        a = self.c1(x)
        b = self.c2(a)
        c = self.c3(b)
        out = torch.cat([a, b, c], dim=1)
        return F.relu(self.bn(out) + s)

# AttentionGate (pour filtrer les skip connections avec l'attention)
class AttentionGate(nn.Module):
    """Attention gate additive : filtre le skip encodeur (x) avec le signal décodeur (g)."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        a = F.relu(self.W_g(g) + self.W_x(x))
        alpha = self.psi(a)          # carte d\'attention dans [0, 1]
        return x * alpha

# Modèle final : Attention MultiResUNet
class AttentionMultiResUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=4, filters=(32, 64, 128, 256, 512)):
        super().__init__()
        f = filters
        # Encodeur
        self.e1 = MultiResBlock(in_ch, f[0])
        self.e2 = MultiResBlock(f[0], f[1])
        self.e3 = MultiResBlock(f[1], f[2])
        self.e4 = MultiResBlock(f[2], f[3])
        self.bottleneck = MultiResBlock(f[3], f[4])
        self.pool = nn.MaxPool2d(2)
        # Décodeur (attention gate + MRB à chaque niveau)
        self.ag4 = AttentionGate(f[4], f[3], f[3] // 2)
        self.d4 = MultiResBlock(f[4] + f[3], f[3])
        self.ag3 = AttentionGate(f[3], f[2], f[2] // 2)
        self.d3 = MultiResBlock(f[3] + f[2], f[2])
        self.ag2 = AttentionGate(f[2], f[1], f[1] // 2)
        self.d2 = MultiResBlock(f[2] + f[1], f[1])
        self.ag1 = AttentionGate(f[1], f[0], f[0] // 2)
        self.d1 = MultiResBlock(f[1] + f[0], f[0])
        # Sortie
        self.out = nn.Conv2d(f[0], num_classes, 1)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))

        g4 = self._up(b)
        d4 = self.d4(torch.cat([g4, self.ag4(g4, s4)], dim=1))
        g3 = self._up(d4)
        d3 = self.d3(torch.cat([g3, self.ag3(g3, s3)], dim=1))
        g2 = self._up(d3)
        d2 = self.d2(torch.cat([g2, self.ag2(g2, s2)], dim=1))
        g1 = self._up(d2)
        d1 = self.d1(torch.cat([g1, self.ag1(g1, s1)], dim=1))
        return self.out(d1)  # logits (N, num_classes, H, W)