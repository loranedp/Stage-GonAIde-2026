# Définition des fonctions pour les statistiques et visualisations
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from utils.segmentation_visualization import (
    CLASS_OVERLAY_COLORS,
    INSTANCE_OVERLAY_COLORS,
    overlay_colored_mask,
    overlay_mask,
    plot_segmentation_comparison,
    tensor_to_numpy_image,
)
from utils.yolo_metrics import mask_to_yolo_polygons


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
    ax=None,
    diagonale=None
):
    # Crée une figure seulement si aucun axe n'est fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        show_plot = True
    else:
        fig = ax.figure
        show_plot = False

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

    # Ligne diagonale rouge en pointillés
    if diagonale:
        ax.axline((0, 0), slope=1, color="red", linestyle="--")

    ax.set_title(titre, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.5)

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
