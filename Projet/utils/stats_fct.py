# Définition des fonctions pour les statistiques et visualisations
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

from utils.segmentation_visualization import (
    CLASS_OVERLAY_COLORS,
    INSTANCE_OVERLAY_COLORS,
    overlay_colored_mask,
    overlay_mask,
    plot_segmentation_comparison,
    tensor_to_numpy_image,
)
from utils.yolo_metrics import mask_to_yolo_polygons


def plot_differences_par_poisson(
    df,
    colonne_predite,
    colonne_annote,
    colonne_id="id poisson",
    ylabel="Valeur prédite − valeur annotée",
    figsize_par_poisson=0.4,
    hauteur=5,
    text_fontsize=None,
    tick_fontsize=8,
):
    """Trace les différences prédites − annotées, triées par poisson.

    Retourne le DataFrame utilisé pour le graphique, ainsi que la figure et
    l'axe Matplotlib, pour permettre de réutiliser ou personnaliser le tracé.
    """
    donnees = df[[colonne_id, colonne_predite, colonne_annote]].dropna(
        subset=[colonne_predite, colonne_annote]
    ).copy()
    donnees["difference"] = donnees[colonne_predite] - donnees[colonne_annote]
    donnees = donnees.sort_values("difference")

    fig, ax = plt.subplots(
        figsize=(max(10, figsize_par_poisson * len(donnees)), hauteur)
    )
    couleurs = np.where(donnees["difference"] < 0, "#4C78A8", "#E45756")
    x = np.arange(len(donnees))
    ax.bar(x, donnees["difference"], color=couleurs)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(donnees[colonne_id].astype(str), rotation=90, fontsize=tick_fontsize)
    ax.set(title="", xlabel="ID poisson", ylabel=ylabel)
    if text_fontsize is not None:
        ax.xaxis.label.set_fontsize(text_fontsize)
        ax.yaxis.label.set_fontsize(text_fontsize)
        ax.tick_params(axis='y', labelsize=text_fontsize)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(handles=[
        Patch(facecolor="#4C78A8", label="Sous-estimation"),
        Patch(facecolor="#E45756", label="Surestimation"),
    ], fontsize=text_fontsize)
    fig.tight_layout()
    plt.show()
    return donnees, fig, ax


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
    diagonale=None,
    text_fontsize=12,
    tick_fontsize=None,
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

    ax.set_title(titre, fontsize=max(14, text_fontsize), pad=15)
    ax.set_xlabel(xlabel, fontsize=text_fontsize)
    ax.set_ylabel(ylabel, fontsize=text_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis='both', labelsize=tick_fontsize)

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



def plot_evolution_curves(df, var1="augmentation_surface_gonade", var2="augmentation_surface_cavite",
                          text_fontsize=None, tick_fontsize=None, legend_fontsize=None):
    """Trace les évolutions et adapte la hauteur au nombre de poissons."""
    poisson_ids = df["id poisson"].dropna().unique()
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Garder une hauteur de tracé lisible tout en donnant une ligne à chaque
    # poisson dans la légende. La légende est ainsi incluse dans la figure.
    hauteur = max(5, 1.5 + 0.28 * (legend_fontsize / 10 if legend_fontsize else 1) * len(poisson_ids))

    for colonne, marqueur, titre in (
        (var1, "o", "Augmentation de la surface des gonades selon la position de l'échographie"),
        (var2, "x", "Augmentation de la surface de la cavité selon la position de l'échographie"),
    ):
        fig, ax = plt.subplots(figsize=(14, hauteur), layout="constrained")
        for i, poisson_id in enumerate(poisson_ids):
            poisson_data = df[df["id poisson"] == poisson_id]
            ax.plot(
                poisson_data["position echo"],
                poisson_data[colonne],
                marker=marqueur,
                color=colors[i % 20],
                label=f"Poisson {poisson_id}",
            )
        ax.set_title(titre, fontsize=text_fontsize)
        ax.set_xlabel("Position de l'échographie", fontsize=text_fontsize)
        ax.set_ylabel("Augmentation de la surface (en %)", fontsize=text_fontsize)
        if tick_fontsize is not None:
            ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
                  fontsize=legend_fontsize)
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
