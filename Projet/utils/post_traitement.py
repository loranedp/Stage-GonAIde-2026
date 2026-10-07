import torch
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def keep_largest_components(preds_tensor):
    preds_np = preds_tensor.byte().cpu().numpy() # Transfert du batch sur le CPU pour traitement
    B, C, H, W = preds_np.shape
    
    for i in range(B):
        # --- 1. Cavité : garde la composante la plus grande ----
        mask = preds_np[i, 0]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8) # Récupère les composantes connectées (adjacents + diagonaux)

        if num_labels > 2: # Si plus de deux composantes, on garde uniquement la plus grande
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            preds_np[i, 0] = (labels == largest_label)

        # --- 2. Gonades : garde au plus deux composantes supérieures à 1000 pixels ---
        mask = preds_np[i, 1]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Garde les composantes dont l'aire est supérieure à 1000 pixels
        kept_labels = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] > 1000) + 1

        # Garde au plus les deux composantes les plus grandes
        if len(kept_labels) > 2:
            top_two = np.argpartition(stats[1:, cv2.CC_STAT_AREA][kept_labels - 1], -2)[-2:]
            kept_labels = kept_labels[top_two]
        preds_np[i, 1] = np.isin(labels, kept_labels).astype(np.uint8)

        # -- 3. Intestin : garde au plus deux composantes supérieures à 300 pixels ---
        if C > 2: # Si la classe intestin est présente
            mask = preds_np[i, 2]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            # Garde au plus deux composantes supérieures à 300 pixels
            kept_labels = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] > 300) + 1
            if len(kept_labels) > 2:
                top_two = np.argpartition(stats[1:, cv2.CC_STAT_AREA][kept_labels - 1], -2)[-2:]
                kept_labels = kept_labels[top_two]
            preds_np[i, 2] = np.isin(labels, kept_labels).astype(np.uint8)

    return torch.from_numpy(preds_np).to(preds_tensor.device).bool()

# Fonction pour corriger les formes incohérentes des masques prédits (nettoyage
# léger et enveloppe convexe pour la cavité, remplissage des trous et suppression
# des excroissances pour les autres classes).
def fill_holes(preds_tensor):
    preds_np = preds_tensor.byte().cpu().numpy() # Transfert du batch sur le CPU pour traitement
    B, C, H, W = preds_np.shape
    
    for i in range(B):
        # --- 1. Cavité : nettoyage léger puis enveloppe convexe ---
        mask = preds_np[i, 0]
        # Supprime les petites excroissances reliées au contour
        cavity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cavity_kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            new_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(new_mask, hull, 1)
            preds_np[i, 0] = new_mask

        # --- 2. Gonades et intestin : remplit les trous et enleve les excroissances ---
        for c in range(1, C):
            mask = preds_np[i, c]

            # Noyau arrondi pour ne pas créer de contours carrés
            if c == 1:  # Gonades
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            if c == 2:  # Intestin
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                
            # Suppression des petites excroissances
            new_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Ouverture morphologique
            # Remplissage des petites encoches sur le contour
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel) # Fermeture morphologique

            # Remplissage des trous internes
            new_mask = binary_fill_holes(new_mask).astype(np.uint8)

            preds_np[i, c] = new_mask

    return torch.from_numpy(preds_np).to(preds_tensor.device).bool()


# Fonction pour séparer les instances d'œufs (sémantique to instance) ainsi que leur appliquer un post-traitement
def split_eggs(preds_tensor, min_distance=10, min_area=100, ouverture_size=3, ouverture_iterations=1):
    """Sépare un masque sémantique d'œufs en instances par watershed : utilise la transformée de distance pour trouver les centres des œufs et
        applique watershed pour séparer les instances.
    Args:
        preds_tensor (torch.Tensor): Masque binaire prédit par le modèle
        min_distance (int): Distance minimale entre deux centres d'œufs pour les considérer comme des instances distinctes
        min_area (int): Aire minimale d'une instance pour être conservée
        ouverture_size (int): Taille du noyau d'ouverture morphologique
        ouverture_iterations (int): Nombre d'itérations d'ouverture morphologique
    """
    # --- Vérification des entrées ---
    if preds_tensor.ndim != 4 or preds_tensor.shape[1] != 1:
        raise ValueError("Le masque des œufs doit avoir la forme (B, 1, H, W).")
    if min_distance < 1 or min_area < 1:
        raise ValueError("min_distance et min_area doivent être strictement positifs.")

    # --- Conversion du tenseur en numpy pour le traitement ---
    preds_np = preds_tensor.detach().byte().cpu().numpy()

     # --- Noyau elliptique pour l'ouverture ---
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ouverture_size, ouverture_size)
    )

    batch_labels = []
    for batch_mask in preds_np:
        binary = batch_mask[0].astype(np.uint8)

        # --- 1. Récupération des composantes connectées ---
        num_components, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, 
            connectivity=8 # Voisinage 8 = pixels adjacents horizontalement, verticalement et diagonalement
        )
        # --- 2.Filtrage des composantes trop petites ---
        clean = np.zeros_like(binary)
        for component_id in range(1, num_components):
            if stats[component_id, cv2.CC_STAT_AREA] >= min_area:
                clean[component_labels == component_id] = 1

        # Si plus de prédictions après le filtrage
        if not clean.any():
            batch_labels.append(np.zeros_like(clean, dtype=np.int32))
            continue

        # --- 3. Transformée de distance pour identifier les centres des œufs ---
        distance = cv2.distanceTransform(clean, cv2.DIST_L2, 5) # calcul pour chaque pixel la distance (L2) au pixel "background" le plus proche

        # --- 4. Identification des maxima locaux -> centre des oeufs ---
        peak_coords = peak_local_max(
            distance,
            min_distance=min_distance, # distance minimale entre deux maxima locaux
            labels=clean,
            exclude_border=False,
        )
        markers = np.zeros_like(clean, dtype=np.int32)
        for marker_id, (row, col) in enumerate(peak_coords, start=1):
            markers[row, col] = marker_id # Associe chaque centre d'œuf à un marqueur unique

        # --- 5. Gestion des composantes connectées qui n'ont pas de marqueur (ex: œufs très proches) ---
        clean_count, clean_components = cv2.connectedComponents(clean, connectivity=8)
        next_marker = int(markers.max()) + 1
        for component_id in range(1, clean_count):
            component = clean_components == component_id
            if not np.any(markers[component]):
                flat_index = np.argmax(np.where(component, distance, -1.0))
                row, col = np.unravel_index(flat_index, distance.shape)
                markers[row, col] = next_marker
                next_marker += 1

        # --- 6. Watershed pour séparer les instances à partir des marqueurs ---
        labels = watershed(-distance, markers, mask=clean).astype(np.int32) # Etend les marqueurs à l'ensemble de la zone binaire, en séparant les instances d'œufs

        # --- 7. Ouverture morphologique pour lisser les contours des instances ---
        opened_labels = np.zeros_like(labels, dtype=np.int32)

        for label_id in np.unique(labels):
            if label_id == 0:
                continue

            # Masque binaire correspondant à un seul œuf
            instance = (labels == label_id).astype(np.uint8)

            opened = cv2.morphologyEx(
                instance,
                cv2.MORPH_OPEN,
                opening_kernel,
                iterations=ouverture_iterations
            )

            opened_labels[opened > 0] = label_id

        # --- 8. Masque final avec filtrage des petites instances potentiellement créées ---
        relabelled = np.zeros_like(labels, dtype=np.int32)
        next_label = 1
        for label_id in np.unique(opened_labels):
            if label_id == 0:
                continue
            instance = opened_labels == label_id
            if int(instance.sum()) < min_area:
                continue
            relabelled[instance] = next_label
            next_label += 1
        batch_labels.append(relabelled)

    return torch.from_numpy(np.stack(batch_labels)).to(preds_tensor.device)
