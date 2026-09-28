import torch

# --------------- Définition des fonctions de perte personnalisées pour l'entraînement du modèle ---------------
# Dice Loss
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) # Transforme les prédictions en probabilités
        intersection = (inputs * targets).sum(dim=(2, 3)) # Intersection entre les prédictions et les cibles
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        return 1. - dice.mean()  # Moyenne sur le batch et les classes
    
# Binary Cross-Entropy Loss + Dice Loss
class BCE_DiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# --------------- Fonctions de pertes avec des pixels ignorés (artefacts) pour l'entraînement du modèle ---------------
# 1. Dice Loss avec prise en compte de ignore_index
class DiceLoss_ignore(torch.nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) # Transforme les prédictions en probabilités

        # --- Ignore des pixels ---
        valid_mask = (targets != self.ignore_index).float() # Masque binaire des pixels valides
        targets_clean = torch.where(targets == self.ignore_index, torch.tensor(0.0, device=targets.device), targets) # Remplace les -100 par 0 pour le calcul de la perte
        inputs_masked = inputs * valid_mask # Neutralise les pixels d'artefacts dans les prédictions
        targets_masked = targets_clean * valid_mask # Neutralise les pixels d'artefacts dans les cibles
        
        # Calcul de l'intersection et de la somme sur les zones valides uniquement
        intersection = (inputs_masked * targets_masked).sum(dim=(2, 3))
        total = inputs_masked.sum(dim=(2, 3)) + targets_masked.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1. - dice.mean()

# 2. BCE + Dice Loss combinée avec ignore_index
class BCE_DiceLoss_ignore(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, ignore_index=-100):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss_ignore(ignore_index=ignore_index)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # --- Ignore des pixels ---
        valid_mask = (targets != self.ignore_index).float() # masque binaire des pixels valides
        targets_clean = torch.where(targets == self.ignore_index, torch.tensor(0.0, device=targets.device), targets) # Remplace les -100 par 0 pour le calcul de la BCE
        
        # Calcul de la BCE pixel par pixel puis moyennage uniquement sur les pixels valides
        bce_raw = self.bce(inputs, targets_clean)
        bce_loss = (bce_raw * valid_mask).sum() / valid_mask.sum().clamp(min=1e-6)
        
        dice_loss = self.dice(inputs, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# Fonction de perte avec régularisation de la longueur des contours
class ContourLengthLoss_ignore(torch.nn.Module):
    def __init__(self, ignore_index=-100, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)

        valid_mask = (targets != self.ignore_index).float()

        # Gradients horizontaux et verticaux
        dx = probs[:, :, :, 1:] - probs[:, :, :, :-1]
        dy = probs[:, :, 1:, :] - probs[:, :, :-1, :]

        # On ignore aussi toute frontière touchant un pixel ignoré
        valid_dx = (
            valid_mask[:, :, :, 1:]
            * valid_mask[:, :, :, :-1]
        )

        valid_dy = (
            valid_mask[:, :, 1:, :]
            * valid_mask[:, :, :-1, :]
        )

        length_x = (
            torch.sqrt(dx.pow(2) + self.eps) * valid_dx
        ).sum() / valid_dx.sum().clamp(min=1.0)

        length_y = (
            torch.sqrt(dy.pow(2) + self.eps) * valid_dy
        ).sum() / valid_dy.sum().clamp(min=1.0)

        return length_x + length_y

class BCE_Dice_ContourLoss_ignore(torch.nn.Module):
    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        contour_weight=0.1,
        ignore_index=-100
    ):
        super().__init__()

        self.bce_dice = BCE_DiceLoss_ignore(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            ignore_index=ignore_index
        )

        self.contour = ContourLengthLoss_ignore(
            ignore_index=ignore_index
        )

        self.contour_weight = contour_weight

    def forward(self, inputs, targets):
        segmentation_loss = self.bce_dice(inputs, targets)
        contour_loss = self.contour(inputs, targets)

        return (
            segmentation_loss
            + self.contour_weight * contour_loss
        )
