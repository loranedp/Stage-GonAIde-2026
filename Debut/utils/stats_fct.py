# Définition des fonctions pour les statistiques et visualisations
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import cv2
from PIL import Image
from matplotlib.colors import BoundaryNorm, ListedColormap, hsv_to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def distribution(
    donnees,
    titre="Distribution",
    xlabel="Valeurs",
    couleur='cornflowerblue',
    bins='auto',
    nombre_graduations_x=None,
    nombre_barres=None,
    ax=None,
    axe_x_fixe=None,
    afficher_courbe=True
):
    """Trace un histogramme avec une courbe de densité facultative.

    afficher_courbe=False masque la courbe de densité.
    couleur accepte les couleurs Matplotlib, notamment "#63A5BF".
    """
    if axe_x_fixe is not None:
        try:
            minimum_x, maximum_x, pas_x = axe_x_fixe
        except (TypeError, ValueError) as erreur:
            raise ValueError(
                "axe_x_fixe doit contenir exactement (minimum, maximum, pas)."
            ) from erreur

        for valeur in (minimum_x, maximum_x, pas_x):
            if (
                isinstance(valeur, (bool, np.bool_))
                or not isinstance(
                    valeur,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(valeur)
            ):
                raise ValueError(
                    "axe_x_fixe doit contenir uniquement des nombres finis."
                )

        if minimum_x >= maximum_x:
            raise ValueError(
                "Le minimum de axe_x_fixe doit être strictement inférieur "
                "au maximum."
            )
        if pas_x <= 0:
            raise ValueError(
                "Le pas de axe_x_fixe doit être strictement positif."
            )

    parametres_entiers = [("nombre_barres", nombre_barres)]
    if axe_x_fixe is None:
        parametres_entiers.append(
            ("nombre_graduations_x", nombre_graduations_x)
        )

    for nom, valeur in parametres_entiers:
        if valeur is not None and (
            isinstance(valeur, (bool, np.bool_))
            or not isinstance(valeur, (int, np.integer))
            or valeur <= 0
        ):
            raise ValueError(f"{nom} doit être un entier strictement positif.")

    if nombre_barres is not None:
        bins_est_automatique = isinstance(bins, str) and bins == 'auto'
        if not bins_est_automatique:
            raise ValueError(
                "Utilisez soit 'bins', soit 'nombre_barres', mais pas les deux."
            )
        bins = nombre_barres

    # Crée une figure uniquement si aucun axe n'est fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        show_plot = True
    else:
        show_plot = False

    sns.histplot(
        donnees,
        kde=afficher_courbe,
        color=couleur,
        bins=bins,
        edgecolor='black',
        alpha=0.8,
        ax=ax
    )

    # Habillage du graphique
    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)

    if axe_x_fixe is not None:
        etendue_x = maximum_x - minimum_x
        nombre_intervalles = etendue_x / pas_x
        entier_proche = round(nombre_intervalles)
        if np.isclose(
            nombre_intervalles,
            entier_proche,
            rtol=1e-12,
            atol=1e-12,
        ):
            nombre_intervalles = entier_proche
        else:
            nombre_intervalles = np.floor(nombre_intervalles)

        graduations_x = minimum_x + pas_x * np.arange(nombre_intervalles + 1)
        if np.isclose(graduations_x[-1], maximum_x, rtol=1e-12, atol=1e-12):
            graduations_x[-1] = maximum_x
        ax.set_xlim(minimum_x, maximum_x)
        ax.set_xticks(graduations_x)
    elif nombre_graduations_x is not None:
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=nombre_graduations_x)
        )

    ax.grid(
        True,
        linestyle='--',
        alpha=0.5,
        axis='y'
    )

    # Affiche seulement si la fonction a créé elle-même la figure
    if show_plot:
        plt.tight_layout()
        plt.show()

    return ax

def nuage_points(
    x,
    y,
    color_var=None,
    titre="Nuage de points",
    xlabel="Axe X",
    ylabel="Axe Y",
    couleur="blue",
    ax=None
):
    # Crée une figure seulement si aucun axe n'est fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        show_plot = True
    else:
        fig = ax.figure
        show_plot = False

    # Si une variable color_var est fournie, on l'utilise pour la couleur
    if color_var is not None:
        scatter = ax.scatter(
            x,
            y,
            c=color_var,
            cmap="viridis",
            alpha=0.8,
            edgecolor="black",
            s=40
        )

        # Barre de couleur associée à la bonne figure / au bon axe
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Position Écho")

    else:
        ax.scatter(
            x,
            y,
            color=couleur,
            alpha=0.8,
            edgecolor="black",
            s=40
        )

    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.5)

    # Si la fonction a créé elle-même la figure, on l'affiche
    if show_plot:
        plt.show()

    return ax


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



def plot_evolution_curves(df, var1 = "augmentation_surface_gonade", var2 = "augmentation_surface_cavite"):
    # Palette de 20 couleurs distinctes
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Graphique 1 : Gonades
    plt.figure(figsize=(10, 5))
    for i, poisson_id in enumerate(df['id poisson'].unique()):
        poisson_data = df[df['id poisson'] == poisson_id]
        plt.plot(
            poisson_data['position echo'],
            poisson_data[var1],
            marker='o',
            color=colors[i % 20],
            label=f"Poisson {poisson_id}"
        )
    plt.title("Augmentation de la surface des gonades selon la position de l'échographie")
    plt.xlabel('Position de l\'échographie')
    plt.ylabel('Augmentation de la surface (en %)')
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
            poisson_data[var2],
            marker='x',
            color=colors[i % 20],
            label=f"Poisson {poisson_id}"
        )
    plt.title("Augmentation de la surface de la cavité selon la position de l'échographie")
    plt.xlabel('Position de l\'échographie')
    plt.ylabel('Augmentation de la surface (en %)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----- Fonctions pour la visualisation des masques et images -----
# Référentiel RGB unique du pipeline cavité : cavité, gonade, intestin.
# Il est partagé par le pré-traitement, les exports de validation et la
# comparaison des modèles.
CLASS_OVERLAY_COLORS = np.array([
    [26, 12, 176],   # Cavité
    [69, 209, 179],  # Gonade
    [199, 182, 179], # Intestin
], dtype=np.uint8)


def tensor_to_numpy_image(tensor):
    if hasattr(tensor, "detach"):
        img = tensor.detach().cpu().numpy()
    else:
        img = np.asarray(tensor)
    if img.ndim == 3:
        # Les tenseurs sont généralement CHW, mais les images numpy peuvent
        # déjà être HWC. Ne pas tronquer par erreur la hauteur d'une image HWC.
        if img.shape[0] <= 4 and img.shape[-1] > 4:
            img = img[:3]
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
        elif img.shape[-1] <= 4:
            img = img[..., :3]
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
    
    overlay = image.copy()
    for c in range(mask.shape[0]):
        color = CLASS_OVERLAY_COLORS[c % len(CLASS_OVERLAY_COLORS)]
        mask_bool = mask[c] > 0.5
        
        for rgb_channel in range(3):
            overlay[..., rgb_channel] = np.where(
                mask_bool,
                (overlay[..., rgb_channel] * (1 - alpha) + color[rgb_channel] * alpha).astype(np.uint8),
                overlay[..., rgb_channel]
            )
    return overlay


# Palette volontairement fixe : elle ne dépend ni de l'ordre d'itération ni
# d'un générateur aléatoire, et reste donc identique d'une exécution à l'autre.
INSTANCE_OVERLAY_COLORS = np.array([
    [230, 70, 60],
    [55, 185, 75],
    [55, 125, 225],
    [205, 65, 175],
    [235, 160, 45],
    [35, 185, 190],
    [150, 75, 215],
    [225, 85, 125],
], dtype=np.uint8)


def overlay_colored_mask(image, mask, alpha=0.45, dataset_name=None,
                         class_names=None, target_size=None):
    """Superpose un masque coloré sur une image.

    Pour ``dataset_name == "oeufs"``, ``mask`` est une carte 2D d'identifiants
    d'instances (ou un masque ``(1, H, W)``). Chaque identifiant positif reçoit
    une couleur déterministe, prolongée au-delà de la palette initiale. Pour les autres
    datasets, ``mask`` est un masque multi-canaux ``(C, H, W)`` et chaque canal
    présent reçoit la couleur fixe de sa classe.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha doit être compris entre 0 et 1.")

    def as_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    base = tensor_to_numpy_image(image)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=2)
    base = np.clip(base, 0, 1) if np.issubdtype(base.dtype, np.floating) else base
    if base.max() <= 1.0:
        base = base * 255
    base = np.clip(base, 0, 255).astype(np.uint8)
    if target_size is not None:
        base = cv2.resize(base, target_size, interpolation=cv2.INTER_LINEAR)

    raw_mask = as_numpy(mask)
    if dataset_name == "oeufs":
        if raw_mask.ndim == 3 and raw_mask.shape[0] == 1:
            raw_mask = raw_mask[0]
        if raw_mask.ndim != 2:
            raise ValueError("Le masque d'instances doit avoir la forme (H, W) ou (1, H, W).")
        labels = raw_mask
    else:
        if raw_mask.ndim == 2:
            raw_mask = raw_mask[None, ...]
        if raw_mask.ndim != 3:
            raise ValueError("Le masque sémantique doit avoir la forme (C, H, W).")
        labels = np.zeros(raw_mask.shape[1:], dtype=np.int32)
        # En cas de chevauchement, le canal de classe le plus élevé est retenu.
        for class_idx, class_mask in enumerate(raw_mask):
            labels[class_mask > 0.5] = class_idx + 1

    if labels.shape != base.shape[:2]:
        labels = cv2.resize(labels.astype(np.int32), (base.shape[1], base.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    labels = np.rint(labels).astype(np.int64)
    output = base.copy()
    colors = INSTANCE_OVERLAY_COLORS if dataset_name == "oeufs" else CLASS_OVERLAY_COLORS
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        if dataset_name == "oeufs" and label_id > len(colors):
            hue = (int(label_id) * 0.618033988749895) % 1.0
            color = np.rint(hsv_to_rgb([hue, 0.75, 0.95]) * 255).astype(np.uint8)
        else:
            color = colors[(int(label_id) - 1) % len(colors)]
        pixels = labels == label_id
        output[pixels] = (
            output[pixels].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    return output


def plot_segmentation_comparison(
    image, mask_true, mask_pred, is_eggs=False, alpha=0.45
):
    """Affiche l'image, la vérité terrain et la prédiction côte à côte.

    En mode standard, les masques sont attendus au format ``(C, H, W)`` avec
    les canaux dans l'ordre suivant : cavité, gonade, intestin. En mode œufs,
    les masques réel et prédit peuvent être des cartes 2D d'identifiants
    d'instances ou des masques binaires au format ``(1, H, W)``.

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
        def egg_mask_to_label_map(mask_tensor, mask_name):
            mask = mask_tensor.detach().cpu().numpy()
            if mask.ndim == 3 and mask.shape[0] == 1:
                return (resize_label_map(mask[0]) > 0.5).astype(np.int32)
            if mask.ndim != 2:
                raise ValueError(
                    f"Le masque {mask_name} des œufs doit avoir la forme "
                    "(1, H, W) ou (H, W)."
                )
            labels = np.rint(resize_label_map(mask)).astype(np.int32)
            if np.any(labels < 0):
                raise ValueError("Les identifiants d'instances doivent être positifs ou nuls.")
            return labels

        def instance_colormap(labels):
            displayed_instances = max(1, int(labels.max()))
            colors = ["#000000"] + [
                plt.cm.tab20(idx % 20) for idx in range(displayed_instances)
            ]
            return (
                ListedColormap(colors),
                BoundaryNorm(
                    np.arange(-0.5, displayed_instances + 1.5), len(colors)
                ),
            )

        true_labels = egg_mask_to_label_map(mask_true, "réel")
        pred_labels = egg_mask_to_label_map(mask_pred, "prédit")
        true_cmap, true_norm = instance_colormap(true_labels)
        pred_cmap, pred_norm = instance_colormap(pred_labels)
        legend_handles = []
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
def mask_to_yolo_polygons(
    mask_bool,
    class_id,
    target_size=(510, 380),
    *,
    largest_only=False,
):
    lines = []

    mask_bool_cpu = mask_bool.cpu().numpy()

    # Redimensionner les masques prédits et réels
    pred_mask_pil = Image.fromarray((mask_bool_cpu > 0.5).astype(np.uint8) * 255)
    pred_mask_resized = pred_mask_pil.resize(target_size, Image.NEAREST)

    mask_np = np.array(pred_mask_resized)
    mask_uint8 = np.ascontiguousarray(mask_np.astype(np.uint8) * 255) # S'assurer que le masque est bien au format attendu par OpenCV
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE permet de garder tous les points du contour
    if largest_only and contours:
        contours = [max(contours, key=cv2.contourArea)]
    
    polygons = []
    h, w = mask_np.shape
    for cnt in contours:
        if len(cnt) >= 3:  # Au moins 3 points pour un polygone
            contour_flat = cnt.reshape(-1, 2)
            poly_str = [f"{pt[0]/w:.6f} {pt[1]/h:.6f}" for pt in contour_flat]
            polygons.append(f"{class_id} " + " ".join(poly_str))

    return polygons



# Affiche une image avec les masques superposés (pour l'instant non utilisée)
def display_image_with_masks(image_path, true_mask_path, pred_mask_path):
    # Charger l'image et les masques
    image = plt.imread(image_path)
    true_mask = plt.imread(true_mask_path)
    pred_mask = plt.imread(pred_mask_path)

    # Créer une figure avec 3 sous-graphes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Afficher l'image originale
    axs[0].imshow(image)
    axs[0].set_title('Image originale')
    axs[0].axis('off')

    # Afficher l'image avec le masque de vérité terrain superposé
    axs[1].imshow(image)
    axs[1].imshow(true_mask, alpha=0.5)  # Superposition avec transparence
    axs[1].set_title('Masque de vérité terrain')
    axs[1].axis('off')

    # Afficher l'image avec le masque prédit superposé
    axs[2].imshow(image)
    axs[2].imshow(pred_mask, alpha=0.5)  # Superposition avec transparence
    axs[2].set_title('Masque prédit')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
