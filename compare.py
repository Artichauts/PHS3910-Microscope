"""
test_objectifs.py

Compare plusieurs objectifs (NA, M) avec les deux méthodes de localisation
(gaussienne et centroïde) et affiche l'erreur de chaque combinaison.

Pour chaque objectif :
    trajectoire brownienne -> images PSF -> localisation -> MSD -> D -> rayon

Les MÊMES trajectoires sont réutilisées pour tous les objectifs, donc les
différences observées viennent de l'optique et de la localisation, pas du
hasard de la trajectoire (avec 50 pas, l'erreur statistique sur D est grande).

Deux métriques par méthode :
    - erreur relative sur le rayon estimé via le MSD (comme dans le script original)
    - erreur de localisation RMS (nm, dans l'espace objet), après retrait du
      décalage constant (un décalage constant n'affecte pas le MSD)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- À adapter à l'arborescence du projet ---------------------------------
from backend.brown_movement import generate_brownian_motion
from backend.fct_PSF import particle_image
from backend.detection_pic import position_gaussian, position_centroid
from backend.regression import regression, msd_fft
# ---------------------------------------------------------------------------


# =========================== PARAMÈTRES ====================================

# Objectifs à tester : (nom, NA, M).  Valeurs d'exemple, à modifier librement.
OBJECTIVES = [
    ("4x / 0.10",  0.10,  -4),
    ("10x / 0.25", 0.25, -10),
    ("10x / 0.45", 0.45, -10),
    ("20x / 0.40", 0.40, -20),
    ("20x / 0.75", 0.75, -20),
    ("40x / 0.65", 0.65, -40),
]

N_REPEATS = 10      # nombre de trajectoires par objectif (1 = un seul essai)
SEED = 0            # graine de base (trajectoire k : SEED + k)
EDGE_MARGIN = 20    # marge (pixels) : la particule doit rester loin des bords

# Mouvement brownien
N_STEPS = 50
R = 2e-6            # rayon de la particule (m)
KB = 1.380649e-23   # constante de Boltzmann (J/K)
T = 298             # température (K)
NU = 1.01e-3        # viscosité de l'eau (Pa.s)
TIME_STEP = 0.05    # durée entre deux images (s)

# Imagerie
WAVELENGTH = 575e-9   # (m)
PIXEL_SIZE = 1.55e-6  # (m)
NY, NX = 507, 380

METHODS = {
    "Gaussien": position_gaussian,
    "Centroïde": position_centroid,
}

# ===========================================================================


def diffusion_coefficient():
    return KB * T / (6 * np.pi * NU * R)


def make_trajectories(D):
    """Une trajectoire par répétition, partagée par tous les objectifs."""
    trajectories = []
    for k in range(N_REPEATS):
        np.random.seed(SEED + k)
        positions_xy, timestamp = generate_brownian_motion(N_STEPS, D, TIME_STEP)
        trajectories.append((np.asarray(positions_xy), np.asarray(timestamp)))
    return trajectories


def render_images(positions_xy, NA, M):
    """Images PSF de la trajectoire. Renvoie None si la particule sort du cadre."""
    Mabs = abs(M)
    x, y = positions_xy[:, 0], positions_xy[:, 1]
    cx = x * Mabs / PIXEL_SIZE + NX / 2
    cy = y * Mabs / PIXEL_SIZE + NY / 2

    if (cx.min() < EDGE_MARGIN or cx.max() > NX - EDGE_MARGIN or
            cy.min() < EDGE_MARGIN or cy.max() > NY - EDGE_MARGIN):
        return None

    images = np.zeros((len(x), NY, NX))
    for i in range(len(x)):
        images[i] = particle_image(NA=NA, M=-Mabs, pixel_size=PIXEL_SIZE, radius=R,
                                   LAMBDA=WAVELENGTH, shape=(NY, NX),
                                   center=(cx[i], cy[i]))
    return images


def analyse_method(localize, images, positions_xy, timestamp, Mabs):
    """Localise, ajuste le MSD, renvoie (erreur relative sur R, RMSE localisation en m)."""
    try:
        # position_gaussian renvoie plus de 2 colonnes (x, y en premier) : on ne garde que x, y
        pos = np.array(localize(images))[:, :2] * PIXEL_SIZE / Mabs

        # Rayon via le MSD (mêmes formules que le script original)
        msd = msd_fft(pos)
        _, slope, _, _ = regression(timestamp, msd)
        D_est = slope / 4
        R_est = KB * T / (6 * np.pi * NU * D_est) if D_est > 0 else np.nan
        err_R = (R_est - R) / R

        # Erreur de localisation (décalage constant retiré)
        true = positions_xy + np.array([NX / 2, NY / 2]) * PIXEL_SIZE / Mabs
        diff = pos - true
        diff = diff - diff.mean(axis=0)
        rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        return err_R, rmse
    except Exception as e:  # ajustement qui échoue (PSF sous-échantillonnée, etc.)
        print(f"  [échec de localisation : {type(e).__name__}: {e}]")
        return np.nan, np.nan


def run_benchmark(trajectories):
    """results[objectif][méthode] = {"err_R": [...], "rmse": [...]}"""
    results = {}
    for name, NA, M in OBJECTIVES:
        Mabs = abs(M)
        results[name] = {m: {"err_R": [], "rmse": []} for m in METHODS}
        results[name]["n_valid"] = 0

        for positions_xy, timestamp in tqdm(trajectories, desc=f"{name:>11}"):
            images = render_images(positions_xy, NA, M)
            if images is None:
                continue  # particule sortie du cadre : essai ignoré
            results[name]["n_valid"] += 1
            for method, localize in METHODS.items():
                err_R, rmse = analyse_method(localize, images, positions_xy, timestamp, Mabs)
                results[name][method]["err_R"].append(err_R)
                results[name][method]["rmse"].append(rmse)
    return results


def summarize(results):
    """Tableau texte + dictionnaire de statistiques pour le graphique."""
    stats = {}
    header = f"{'Objectif':<12}{'NA':>6}{'M':>5}{'n':>4}"
    for m in METHODS:
        header += f" | {m + ' : err R (%)':>24}{'RMSE loc (nm)':>15}"
    print("\n" + header)
    print("-" * len(header))

    for name, NA, M in OBJECTIVES:
        r = results[name]
        row = f"{name:<12}{NA:>6.2f}{M:>5d}{r['n_valid']:>4d}"
        stats[name] = {}
        for m in METHODS:
            err = np.abs(np.array(r[m]["err_R"], dtype=float)) * 100
            rmse = np.array(r[m]["rmse"], dtype=float) * 1e9
            if r["n_valid"] == 0 or np.all(np.isnan(err)):
                row += f" | {'n/a':>24}{'n/a':>15}"
                stats[name][m] = (np.nan, np.nan, np.nan)
                continue
            mean_e, std_e = np.nanmean(err), np.nanstd(err)
            mean_r = np.nanmean(rmse)
            stats[name][m] = (mean_e, std_e, mean_r)
            row += f" | {f'{mean_e:.2f} ± {std_e:.2f}':>24}{mean_r:>15.1f}"
        print(row)

    print("\nerr R = |R estimé - R vrai| / R vrai, moyenne ± écart-type sur les trajectoires.")
    return stats


def plot_results(stats):
    names = [n for n, _, _ in OBJECTIVES]
    xs = np.arange(len(names))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for j, m in enumerate(METHODS):
        offset = (j - 0.5) * width
        mean_e = [stats[n][m][0] for n in names]
        std_e = [stats[n][m][1] for n in names]
        rmse = [stats[n][m][2] for n in names]
        axes[0].bar(xs + offset, mean_e, width, yerr=std_e, capsize=3, label=m)
        axes[1].bar(xs + offset, rmse, width, label=m)

    for ax, ylabel, title in zip(
            axes,
            ["Erreur relative sur le rayon (%)", "Erreur de localisation RMS (nm)"],
            ["Erreur sur le rayon (via MSD)", "Précision de localisation"]):
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    D = diffusion_coefficient()
    print(f"Coefficient de diffusion théorique : {D:.4e} m^2/s")
    print(f"{len(OBJECTIVES)} objectifs x {N_REPEATS} trajectoires x {N_STEPS} images "
          f"({NY}x{NX}) x {len(METHODS)} méthodes")

    trajectories = make_trajectories(D)
    results = run_benchmark(trajectories)
    stats = summarize(results)
    plot_results(stats)