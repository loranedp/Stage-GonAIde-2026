import torch
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

# ---------------- Post-traitement : conserve un seul masque par classe ----------------
def keep_largest_components(preds_tensor):
    preds_np = preds_tensor.byte().cpu().numpy() # Transfert du batch sur le CPU pour traitement
    B, C, H, W = preds_np.shape
    
    for i in range(B):
        # --- 1. Cavité : garde la composante la plus grande ----
        mask = preds_np[i, 0]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # Récupère les composantes connectées

        if num_labels > 2: # Si plus de deux composantes (fond + au moins 2 objets), on garde la plus grande
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            preds_np[i, 0] = (labels == largest_label)

        # --- 2. Gonades : garde toutes les composantes supérieures à 1000 pixels ---
        mask = preds_np[i, 1]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 2: # Si plus de deux composantes (fond + au moins 2 objets), on garde toutes celles supérieure à 1000 pixels
            large_components = stats[1:, cv2.CC_STAT_AREA] > 1000
            preds_np[i, 1] = np.isin(labels, np.where(large_components)[0] + 1).astype(np.uint8) # +1 car on ignore le label 0 (fond)

        # -- 3. Intestin : garde toutes les composantes supérieures à 300 pixels ---
        mask = preds_np[i, 2]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels > 2: # Si plus de deux composantes (fond + au moins 2 objets), on garde toutes celles supérieure à 300 pixels
            large_components = stats[1:, cv2.CC_STAT_AREA] > 300
            preds_np[i, 2] = np.isin(labels, np.where(large_components)[0] + 1).astype(np.uint8) # +1 car on ignore le label 0 (fond)

    return torch.from_numpy(preds_np).to(preds_tensor.device).bool()

# Fonction pour corriger les formes incohérentes des masques prédits (remplissage des trous et enveloppe convexe pour la cavité)
def fill_holes(preds_tensor):
    preds_np = preds_tensor.byte().cpu().numpy() # Transfert du batch sur le CPU pour traitement
    B, C, H, W = preds_np.shape
    
    for i in range(B):
        # --- 1. Cavité : enveloppe convexe ---
        mask = preds_np[i, 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            new_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(new_mask, hull, 1)
            preds_np[i, 0] = new_mask

        # --- 2. Gonades et intestin : remplit les trous ---
        for c in [1, 2]:
            mask = preds_np[i, c]
            new_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
            new_mask = binary_fill_holes(new_mask).astype(np.uint8)
            preds_np[i, c] = new_mask

    return torch.from_numpy(preds_np).to(preds_tensor.device).bool()