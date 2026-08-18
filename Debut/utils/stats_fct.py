# Définition des fonctions pour les statistiques et visualisations
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import cv2
from PIL import Image
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def distribution(donnees, titre="Distribution", xlabel="Valeurs", couleur='cornflowerblue'):
    plt.figure(figsize=(6, 4))
    
    sns.histplot(
        donnees, 
        kde=True,             # Ajoute la densité de probabilité
        color=couleur, 
        bins='auto',          # intervalles automatiques
        edgecolor='black',    # Bordure des colonnes
        alpha=0.8
    )
    
    # Habillage du graphique
    plt.title(titre, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Fréquence (Nombre d'observations)", fontsize=12)
    
    # Grille horizontale uniquement pour faciliter la lecture des hauteurs
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Affichage
    plt.show()

def nuage_points(x, y, color_var=None, titre="Nuage de points", xlabel="Axe X", ylabel="Axe Y", couleur='blue'):
    plt.figure(figsize=(6, 4))

    # Si une variable 'color_var' est fournie, on l'utilise pour la couleur
    if color_var is not None:
        scatter = plt.scatter(x, y, 
                                c=color_var,     # La variable pour colorier les points
                                cmap='viridis',      # La palette de couleurs
                                alpha=0.8,           # Transparence des points
                                edgecolor='black',   # Bordure des points
                                s=40                 # Taille des points
                                )
        # Ajout d'une barre de légende pour les couleurs
        cbar = plt.colorbar(scatter)
        cbar.set_label('Position Écho')
    else:
        # Comportement par défaut si 'color_var' n'est pas renseigné
        plt.scatter(x, y, color=couleur, alpha=0.8, edgecolor='black', s=40)

    # Ajout des éléments de texte
    plt.title(titre, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Ajout d'une grille discrète
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Affichage du graphique
    plt.show()


def linear_regression(x, y):
    # Ajout d'une constante pour le modèle
    x_cst = sm.add_constant(x)
    
    # ---- Création et ajustement du modèle OLS ----
    model = sm.OLS(y, x_cst).fit()
    residus = model.resid

    # ---- Création de la figure côte à côte (les 2 graphiques) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ==========================================
    # Graphique 1 : Résidus vs Valeurs Prédites
    # ==========================================
    sns.residplot(
        x=model.fittedvalues, # Valeurs prédites par le modèle
        y=residus,             # Résidus
        lowess=True,           # Ajoute une ligne de tendance lissée
        ax=ax1,                # On force le tracé sur le 1er axe
        line_kws={'color': 'red', 'lw': 2}, # Style de la ligne de tendance
        scatter_kws={'alpha': 0.7, 'edgecolor': 'black'}
    )

    ax1.set_title("Résidus vs Valeurs Prédites", fontsize=14, pad=15)
    ax1.set_xlabel("Valeurs Prédites (Fitted values)")
    ax1.set_ylabel("Résidus")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # Graphique 2 : Influence Plot
    # ==========================================
    # 1. Extraction des données d'influence
    influence = model.get_influence()
    cooks_d, _ = influence.cooks_distance
    leverage = influence.hat_matrix_diag
    studentized_resids = influence.resid_studentized_external

    # On récupère les indices des 5 plus grandes distances de Cook
    top_5_idx = np.argsort(cooks_d)[-5:]

    # Tracé du graphique sur le 2ème axe
    sm.graphics.influence_plot(model, ax=ax2, criterion="cooks", alpha=0.20)

    # 2. Suppression de tous les labels générés automatiquement par statsmodels
    for txt in ax2.texts:
        txt.remove()

    # 3. Ajout manuel des labels uniquement pour nos 5 plus gros points
    labels_originaux = model.model.data.row_labels
    for i in top_5_idx:
        x_pos = leverage[i]
        y_pos = studentized_resids[i]
        nom_label = labels_originaux[i]
        
        ax2.text(x_pos, y_pos, str(nom_label), 
                fontsize=10, 
                color='firebrick', 
                fontweight='bold',
                ha='right',   # Alignement horizontal
                va='bottom')  # Alignement vertical

    # Habillage du 2ème graphique
    ax2.set_title("Influence Plot", fontsize=14, pad=15)
    ax2.set_xlabel("H Leverage (Influence de la position X)", fontsize=12)
    ax2.set_ylabel("Résidus Studentisés (Taille de l'erreur)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return model


def plot_evolution_curves(df):
    # Palette de 20 couleurs distinctes
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Graphique 1 : Gonades
    plt.figure(figsize=(10, 5))
    for i, poisson_id in enumerate(df['id poisson'].unique()):
        poisson_data = df[df['id poisson'] == poisson_id]
        plt.plot(
            poisson_data['position echo'],
            poisson_data['augmentation_surface_gonade'],
            marker='o',
            color=colors[i % 20],
            label=f"Poisson {poisson_id}"
        )
    plt.title("Augmentation de la surface des gonades par position d'échographie")
    plt.xlabel('Position d\'échographie')
    plt.ylabel('Augmentation de la surface (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Graphique 2 : Cavités
    plt.figure(figsize=(10, 5))
    for i, poisson_id in enumerate(df['id poisson'].unique()):
        poisson_data = df[df['id poisson'] == poisson_id]
        plt.plot(
            poisson_data['position echo'],
            poisson_data['augmentation_surface_cavite'],
            marker='x',
            color=colors[i % 20],
            label=f"Poisson {poisson_id}"
        )
    plt.title("Augmentation de la surface de la cavité par position d'échographie")
    plt.xlabel('Position d\'échographie')
    plt.ylabel('Augmentation de la surface (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----- Fonctions pour la visualisation des masques et images -----
def tensor_to_numpy_image(tensor):
    img = tensor.cpu().numpy()
    if img.ndim == 3 and img.shape[0] > 3: # Conserve que les 3 premiers canaux si plus de 3
        img = img[:3]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.max() > 1.0:
        img = img / 255.0
    return img

def overlay_mask(image_tensor, mask_tensor, alpha=0.4, target_size=None):
    # On récupère l'image en format 0-1
    image = tensor_to_numpy_image(image_tensor)
    # Conversion de l'image en 0-255
    image = (image * 255).astype(np.uint8)
    
    mask = mask_tensor.cpu().numpy().astype(np.float32)

    if target_size is not None:
        image = cv2.resize(image, target_size) # Redimensionnement

    h, w = image.shape[:2]
    num_mask_classes = mask.shape[0]
    resized_mask = np.zeros((num_mask_classes, h, w), dtype=np.float32)
    for c in range(num_mask_classes):
        if mask[c].shape != (h, w):
            resized_mask[c] = cv2.resize(mask[c], (w, h))
        else:
            resized_mask[c] = mask[c]
            
    mask = resized_mask
    
    # Couleurs en format 0-255 (R, V, B)
    colors = [
        [26, 12, 176],   # Classe 1 : Cavite
        [69, 209, 179],  # Classe 2 : Gonade
        [199, 182, 179], # Classe 3 : Intestin
    ]
    
    overlay = image.copy()
    for c in range(mask.shape[0]):
        color = colors[c % len(colors)]
        mask_bool = mask[c] > 0.5
        
        for rgb_channel in range(3):
            overlay[..., rgb_channel] = np.where(
                mask_bool,
                (overlay[..., rgb_channel] * (1 - alpha) + color[rgb_channel] * alpha).astype(np.uint8),
                overlay[..., rgb_channel]
            )
    return overlay


def plot_segmentation_comparison(
    image, mask_true, mask_pred, is_eggs=False, alpha=0.45
):
    """Affiche l'image, la vérité terrain et la prédiction côte à côte.

    En mode standard, les masques sont attendus au format ``(C, H, W)`` avec
    les canaux dans l'ordre suivant : cavité, gonade, intestin. En mode œufs,
    le masque prédit peut être une carte 2D d'identifiants d'instances ou un
    masque binaire au format ``(1, H, W)``.

    Args:
        image: Tenseur image au format ``(C, H, W)``.
        mask_true: Tenseur du masque réel au format ``(C, H, W)``.
        mask_pred: Masque prédit multi-canal, binaire ou d'instances.
        is_eggs: Active l'affichage spécifique aux instances d'œufs.
        alpha: Transparence des masques superposés, entre 0 et 1.

    Returns:
        Un tuple ``(fig, axes)`` contenant la figure et ses trois axes.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha doit être compris entre 0 et 1.")

    image_display = np.clip(tensor_to_numpy_image(image), 0, 1)
    height, width = image_display.shape[:2]

    def resize_label_map(label_map):
        if label_map.shape != (height, width):
            label_map = cv2.resize(
                label_map.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        return label_map

    def semantic_mask_to_label_map(mask_tensor):
        mask = mask_tensor.detach().cpu().numpy()
        if mask.ndim != 3:
            raise ValueError("Les masques doivent avoir la forme (C, H, W).")
        if not 1 <= mask.shape[0] <= 3:
            raise ValueError("Les masques doivent contenir entre 1 et 3 classes.")

        label_map = np.zeros((height, width), dtype=np.uint8)
        for class_idx, class_mask in enumerate(mask):
            class_mask = resize_label_map(class_mask)
            label_map[class_mask > 0.5] = class_idx + 1
        return label_map

    if is_eggs:
        true_mask = mask_true.detach().cpu().numpy()
        if true_mask.ndim == 3 and true_mask.shape[0] == 1:
            true_mask = true_mask[0]
        elif true_mask.ndim != 2:
            raise ValueError("Le masque réel des œufs doit avoir la forme (1, H, W) ou (H, W).")
        true_labels = (resize_label_map(true_mask) > 0.5).astype(np.uint8)

        pred_mask = mask_pred.detach().cpu().numpy()
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_labels = (
                resize_label_map(pred_mask[0]) > 0.5
            ).astype(np.int32)
        elif pred_mask.ndim == 2:
            pred_labels = np.rint(resize_label_map(pred_mask)).astype(np.int32)
            if np.any(pred_labels < 0):
                raise ValueError("Les identifiants d'instances doivent être positifs ou nuls.")
        else:
            raise ValueError(
                "Le masque prédit des œufs doit avoir la forme (1, H, W) ou (H, W)."
            )

        instance_count = int(pred_labels.max())
        true_cmap = ListedColormap(["#000000", plt.cm.tab10(0)])
        true_norm = BoundaryNorm(np.arange(-0.5, 2.5), true_cmap.N)
        displayed_instances = max(1, instance_count)
        pred_colors = ["#000000"] + [
            plt.cm.tab20(idx % 20) for idx in range(displayed_instances)
        ]
        pred_cmap = ListedColormap(pred_colors)
        pred_norm = BoundaryNorm(
            np.arange(-0.5, displayed_instances + 1.5), pred_cmap.N
        )
        legend_handles = [
            Patch(color=pred_colors[idx], label=f"Instance {idx}")
            for idx in range(1, instance_count + 1)
        ]
    else:
        class_names = ["Cavité", "Gonade", "Intestin"]
        class_colors = [
            "#1A0CB0",  # Cavité : bleu foncé
            "#66CCFF",  # Gonade : bleu clair
            "#8E44AD",  # Intestin : violet
        ]
        if mask_true.shape[0] != mask_pred.shape[0]:
            raise ValueError("mask_true et mask_pred doivent avoir le même nombre de classes.")

        true_labels = semantic_mask_to_label_map(mask_true)
        pred_labels = semantic_mask_to_label_map(mask_pred)
        semantic_cmap = ListedColormap(["#000000", *class_colors])
        semantic_norm = BoundaryNorm(
            np.arange(-0.5, len(class_colors) + 1.5), semantic_cmap.N
        )
        true_cmap = pred_cmap = semantic_cmap
        true_norm = pred_norm = semantic_norm
        legend_handles = [
            Patch(color=class_colors[idx], label=class_names[idx])
            for idx in range(mask_true.shape[0])
        ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    titles = ["Image originale", "Vérité terrain", "Prédiction"]
    overlays = [None, true_labels, pred_labels]
    colormaps = [None, true_cmap, pred_cmap]
    norms = [None, true_norm, pred_norm]

    for ax, title, labels, cmap, norm in zip(
        axes, titles, overlays, colormaps, norms
    ):
        ax.imshow(image_display)
        if labels is not None:
            masked_labels = np.ma.masked_where(labels == 0, labels)
            ax.imshow(masked_labels, cmap=cmap, norm=norm, alpha=alpha)
        ax.set_title(title)
        ax.axis("off")

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(5, len(legend_handles)),
        )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()
    return fig, axes

# ----- Fonction pour extraire et formater en YOLO les contours d'un masque COCO ----
def mask_to_yolo_polygons(mask_bool, class_id, target_size=(640, 480)):
    lines = []

    mask_bool_cpu = mask_bool.cpu().numpy()

    # Redimensionner les masques prédits et réels
    pred_mask_pil = Image.fromarray((mask_bool_cpu > 0.5).astype(np.uint8) * 255)
    pred_mask_resized = pred_mask_pil.resize(target_size, Image.NEAREST)

    mask_np = np.array(pred_mask_resized)
    mask_uint8 = np.ascontiguousarray(mask_np.astype(np.uint8) * 255) # S'assurer que le masque est bien au format attendu par OpenCV
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE permet de garder tous les points du contour
    
    polygons = []
    h, w = mask_np.shape
    for cnt in contours:
        if len(cnt) >= 3:  # Au moins 3 points pour un polygone
            contour_flat = cnt.reshape(-1, 2)
            poly_str = [f"{pt[0]/w:.6f} {pt[1]/h:.6f}" for pt in contour_flat]
            polygons.append(f"{class_id} " + " ".join(poly_str))

    return polygons
