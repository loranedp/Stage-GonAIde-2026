# Définition des fonctions pour les statistiques et visualisations
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import cv2
from PIL import Image


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