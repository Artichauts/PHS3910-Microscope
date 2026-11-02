"""Simulation d'un systeme complet de suivi de particules browniennes.

Ce script compare les trajectoires reconstruites par suivre_particules()
aux trajectoires reelles connues de la simulation.
"""

import math
import numpy as np

from backend.fct_PSF import particle_image
from backend.detection_pic_revise import suivre_particules


# =============================================================================
# 1. PARAMETRES A MODIFIER
# =============================================================================

# Dimensions de la camera Raspberry Pi HQ SC1220.
# Pour numpy, l'ordre est (hauteur, largeur).
NY = 3040
NX = 4056

# Taille d'un pixel physique sur le capteur.
PIXEL_SIZE_M = 1.55e-6

# Longueur d'onde utilisee dans le modele de PSF.
LAMBDA_M = 575e-9

# Proprietes du liquide.
KB = 1.380649e-23
TEMPERATURE_K = 298.0
VISCOSITY_PA_S = 1.01e-3

# Rayon de particule utilise pour le premier essai.
PARTICLE_RADII_M = [2.0e-6]

# Nombre de particules simulees dans chaque sequence.
N_PARTICLES = 5

# Nombre d'images par sequence.
N_STEPS = 10

# Nombre de sequences independantes pour chaque configuration.
N_REPETITIONS = 3

# Intensite relative des particules. La premiere est la plus lumineuse.
PARTICLE_INTENSITIES = [1.0, 0.8, 0.65, 0.5, 0.35]

# Bruit photonique de Poisson.
# Le SNR est defini au pixel le plus lumineux de la particule la plus intense.
SNR_DB = 10.0
BACKGROUND_PHOTONS = 5.0

# Parametres de detection de la premiere image.
K_THRESHOLD = 5
MIN_AREA = 5

# Tolerance physique commune pour tous les objectifs.
TOLERANCE_UM = 0.5
TOLERANCE_M = TOLERANCE_UM * 1e-6

# Limite de deplacement pour empecher un fit de sauter sur une autre tache.
MAX_STEP_SIGMAS = 5.0

# Parametres de fenetre du suivi.
TRACK_MARGIN_PX = 10
MINIMUM_FIT_SNR = 1.5

# Les memes graines de base sont reutilisees entre configurations pour 
# faciliter la comparaison des essais.
RANDOM_SEED = 12345

# Configurations : (grossissement, ouverture numerique, intervalle entre images).
CONFIGURATIONS = [
    (4, 0.10, 6.95),
    (10, 0.25, 1.11),
    (20, 0.40, 0.278),
    (40, 0.70, 0.0695),
    (60, 0.85, 0.0309),
]


# =============================================================================
# 2. FONCTIONS UTILITAIRES
# =============================================================================

def calculer_diffusion(radius_m):
    """Calcule D avec la relation de Stokes-Einstein."""

    diffusion = KB * TEMPERATURE_K / (
        6.0 * math.pi * VISCOSITY_PA_S * radius_m
    )

    return diffusion


def tolerance_en_pixels(tolerance_m, magnification):
    """Convertit une tolerance physique fixe en pixels pour un objectif.

    Formule : tolerance_px = tolerance_m * |M| / taille_pixel.
    """

    tolerance_px = (
        tolerance_m * abs(magnification) / PIXEL_SIZE_M
    )

    return tolerance_px


def calculer_demi_fenetre(radius_m, magnification, NA):
    """Estime une fenetre assez grande pour contenir l'image d'une particule.

    L'estimation inclut le rayon apparent de la particule, plusieurs rayons
    d'Airy et une marge. Elle sert au fit initial de suivi.
    """

    rayon_pixels = radius_m * abs(magnification) / PIXEL_SIZE_M

    rayon_airy_pixels = (
        0.61 * LAMBDA_M * abs(magnification)
        / (NA * PIXEL_SIZE_M)
    )

    demi_fenetre = int(
        math.ceil(rayon_pixels + 3.0 * rayon_airy_pixels + 10.0)
    )

    return max(20, demi_fenetre)


def generer_trajectoire(N_steps, diffusion, time_step, rng):
    """Genere une trajectoire 2D brownienne en metres."""

    positions = np.zeros((N_steps, 2), dtype=float)

    ecart_type = math.sqrt(2.0 * diffusion * time_step)

    for image_index in range(1, N_steps):
        deplacement_x = rng.normal(0.0, ecart_type)
        deplacement_y = rng.normal(0.0, ecart_type)

        positions[image_index, 0] = (
            positions[image_index - 1, 0] + deplacement_x
        )

        positions[image_index, 1] = (
            positions[image_index - 1, 1] + deplacement_y
        )

    return positions


def generer_trajectoires_pixels(
    diffusion,
    time_step,
    magnification,
    rng
):
    """Genere les positions reelles des particules en pixels.

    Les positions initiales sont reparties dans l'image. Les deplacements
    brownien physiques sont ensuite convertis en pixels du capteur.

    Retour : tableau (N_STEPS, N_PARTICLES, 2).
    """

    centres_initiaux = [
        (0.18 * NX, 0.25 * NY),
        (0.50 * NX, 0.22 * NY),
        (0.82 * NX, 0.25 * NY),
        (0.33 * NX, 0.75 * NY),
        (0.68 * NX, 0.75 * NY),
    ]

    trajectoires = np.zeros(
        (N_STEPS, N_PARTICLES, 2),
        dtype=float
    )

    for particle_index in range(N_PARTICLES):
        trajectoire_metres = generer_trajectoire(
            N_STEPS,
            diffusion,
            time_step,
            rng
        )

        x_initial, y_initial = centres_initiaux[particle_index]

        for image_index in range(N_STEPS):
            deplacement_x_px = (
                trajectoire_metres[image_index, 0]
                * abs(magnification)
                / PIXEL_SIZE_M
            )

            deplacement_y_px = (
                trajectoire_metres[image_index, 1]
                * abs(magnification)
                / PIXEL_SIZE_M
            )

            trajectoires[image_index, particle_index, 0] = (
                x_initial + deplacement_x_px
            )

            trajectoires[image_index, particle_index, 1] = (
                y_initial + deplacement_y_px
            )

    return trajectoires


def generer_image_propre(positions, magnification, NA, radius_m):
    """Genere une image sans bruit avec toutes les particules de la sequence."""

    image = np.zeros((NY, NX), dtype=np.float32)

    rayon_pixels = radius_m * abs(magnification) / PIXEL_SIZE_M

    rayon_airy_pixels = (
        0.61 * LAMBDA_M * abs(magnification)
        / (NA * PIXEL_SIZE_M)
    )

    demi_largeur = int(
        math.ceil(rayon_pixels + 3.0 * rayon_airy_pixels + 10.0)
    )

    demi_largeur = max(40, demi_largeur)

    for particle_index in range(N_PARTICLES):
        x_centre = float(positions[particle_index, 0])
        y_centre = float(positions[particle_index, 1])

        x1 = max(int(math.floor(x_centre)) - demi_largeur, 0)
        x2 = min(int(math.floor(x_centre)) + demi_largeur + 1, NX)

        y1 = max(int(math.floor(y_centre)) - demi_largeur, 0)
        y2 = min(int(math.floor(y_centre)) + demi_largeur + 1, NY)

        centre_local = (x_centre - x1, y_centre - y1)
        forme_locale = (y2 - y1, x2 - x1)

        image_particule = particle_image(
            NA=NA,
            M=-abs(magnification),
            pixel_size=PIXEL_SIZE_M,
            radius=radius_m,
            LAMBDA=LAMBDA_M,
            shape=forme_locale,
            center=centre_local
        )

        intensite = PARTICLE_INTENSITIES[particle_index]

        image[y1:y2, x1:x2] += (
            intensite * np.asarray(image_particule, dtype=np.float32)
        )

    image = np.maximum(image, 0.0)

    return image


def ajouter_bruit_poisson(image_propre, snr_db, fond_photons, rng):
    """Convertit l'image en photons et ajoute un bruit de Poisson.

    Le facteur d'echelle est choisi pour que le pixel le plus lumineux
    de l'image ait le SNR demande :
        SNR = signal / sqrt(signal + fond).

    Le SNR est exprime en dB avec 20*log10(SNR).
    """

    maximum_signal = float(np.max(image_propre))

    if maximum_signal <= 0.0:
        raise ValueError("L'image propre ne contient aucun signal positif.")

    snr_lineaire = 10.0 ** (snr_db / 20.0)
    snr_carre = snr_lineaire ** 2

    signal_max_photons = 0.5 * (
        snr_carre
        + math.sqrt(
            snr_carre ** 2
            + 4.0 * snr_carre * fond_photons
        )
    )

    facteur_photons = signal_max_photons / maximum_signal

    photons_moyens = (
        fond_photons
        + facteur_photons * image_propre
    )

    image_bruitee = rng.poisson(photons_moyens)

    return image_bruitee.astype(np.float32)


# =============================================================================
# 3. ASSOCIATION DES TRAJECTOIRES ET EVALUATION
# =============================================================================

def position_est_valide(position):
    """Indique si une position (x, y) contient deux nombres finis."""

    if position is None or len(position) < 2:
        return False

    return bool(
        np.isfinite(position[0]) and np.isfinite(position[1])
    )


def associer_traces_initiales(traces, positions_reelles, tolerance_px):
    """Associe chaque trace initiale a au plus une particule reelle.

    On construit toutes les distances trace-particule, puis on les trie.
    On associe d'abord les paires les plus proches, sans reutiliser une
    trace ou une particule. C'est une association gloutonne simple.

    Retour : dictionnaire {indice_trace: indice_particule_reelle}.
    """

    distances = []

    for track_index in range(len(traces)):
        if len(traces[track_index]) == 0:
            continue

        position_initiale = traces[track_index][0]

        if not position_est_valide(position_initiale):
            continue

        for particle_index in range(len(positions_reelles)):
            x_reel, y_reel = positions_reelles[particle_index]

            dx = position_initiale[0] - x_reel
            dy = position_initiale[1] - y_reel
            distance = math.sqrt(dx * dx + dy * dy)

            distances.append((distance, track_index, particle_index))

    distances.sort(key=lambda element: element[0])

    traces_utilisees = set()
    particules_utilisees = set()
    associations = {}

    for distance, track_index, particle_index in distances:
        if distance > tolerance_px:
            break

        if track_index in traces_utilisees:
            continue

        if particle_index in particules_utilisees:
            continue

        associations[track_index] = particle_index
        traces_utilisees.add(track_index)
        particules_utilisees.add(particle_index)

    return associations


def evaluer_suivi(traces, positions_reelles, tolerance_px):
    """Calcule TP, FP et FN en respectant l'identite des trajectoires.

    Une trace est associee a une particule a partir de la premiere image.
    Elle doit ensuite rester a moins de tolerance_px de CETTE particule.
    Si elle saute sur une autre particule, elle n'est pas consideree comme
    un suivi correct : on compte un FP et un FN pour cette image.
    """

    nombre_images = positions_reelles.shape[0]
    nombre_particules = positions_reelles.shape[1]

    associations = associer_traces_initiales(
        traces,
        positions_reelles[0],
        tolerance_px
    )

    trace_par_particule = {}

    for track_index, particle_index in associations.items():
        trace_par_particule[particle_index] = track_index

    traces_associees = set(associations.keys())
    traces_non_associees = []

    for track_index in range(len(traces)):
        if track_index not in traces_associees:
            traces_non_associees.append(track_index)

    vrais_positifs = 0
    faux_positifs = 0
    faux_negatifs = 0

    erreurs_localisation_px = []
    resultats_par_image = []

    for image_index in range(nombre_images):
        tp_image = 0
        fp_image = 0
        fn_image = 0

        # Evaluer une fois chacune des particules reelles.
        for particle_index in range(nombre_particules):
            if particle_index not in trace_par_particule:
                fn_image += 1
                continue

            track_index = trace_par_particule[particle_index]

            if image_index >= len(traces[track_index]):
                fn_image += 1
                continue

            position_estimee = traces[track_index][image_index]

            if not position_est_valide(position_estimee):
                fn_image += 1
                continue

            position_reelle = positions_reelles[
                image_index,
                particle_index
            ]

            dx = position_estimee[0] - position_reelle[0]
            dy = position_estimee[1] - position_reelle[1]
            erreur = math.sqrt(dx * dx + dy * dy)

            erreurs_localisation_px.append(erreur)

            if erreur <= tolerance_px:
                tp_image += 1
            else:
                # Une position estimee trop loin est un faux positif ;
                # la vraie position de la particule est simultanement manquee.
                fp_image += 1
                fn_image += 1

        # Les traces initiales qui n'ont ete associees a aucune vraie
        # particule sont des fausses pistes. Chaque position finie produite
        # par ces pistes est comptee comme un faux positif.
        for track_index in traces_non_associees:
            if image_index >= len(traces[track_index]):
                continue

            position = traces[track_index][image_index]

            if position_est_valide(position):
                fp_image += 1

        vrais_positifs += tp_image
        faux_positifs += fp_image
        faux_negatifs += fn_image

        resultats_par_image.append({
            "image": image_index + 1,
            "TP": tp_image,
            "FP": fp_image,
            "FN": fn_image,
        })

    nombre_positions_reelles = nombre_images * nombre_particules

    if vrais_positifs + faux_positifs > 0:
        precision = vrais_positifs / (vrais_positifs + faux_positifs)
    else:
        precision = 0.0

    if vrais_positifs + faux_negatifs > 0:
        rappel = vrais_positifs / (vrais_positifs + faux_negatifs)
    else:
        rappel = 0.0

    if precision + rappel > 0.0:
        score_f1 = 2.0 * precision * rappel / (precision + rappel)
    else:
        score_f1 = 0.0

    if len(erreurs_localisation_px) > 0:
        rmse_px = math.sqrt(
            float(np.mean(np.square(erreurs_localisation_px)))
        )
    else:
        rmse_px = float("nan")

    nombre_traces_associees = len(associations)
    nombre_fausses_traces_initiales = len(traces_non_associees)

    taux_initialisation = (
        nombre_traces_associees / nombre_particules
        if nombre_particules > 0 else 0.0
    )

    rappel = vrais_positifs / nombre_positions_reelles
    # Le rappel ci-dessus est egal a TP/(TP+FN), car chaque position reelle
    # est soit un TP soit un FN.

    return {
        "TP": vrais_positifs,
        "FP": faux_positifs,
        "FN": faux_negatifs,
        "precision": precision,
        "rappel": rappel,
        "F1": score_f1,
        "RMSE_px": rmse_px,
        "traces_initiales": len(traces),
        "traces_associees": nombre_traces_associees,
        "fausses_traces_initiales": nombre_fausses_traces_initiales,
        "taux_initialisation": taux_initialisation,
        "par_image": resultats_par_image,
    }


def verifier_separation(
    trajectoires_reelles,
    tolerance_px,
    radius_m,
    magnification
):
    """Verifie la separation minimale entre particules pendant la sequence.

    Les distances sont d'abord calculees en pixels, puis converties en metres
    dans le plan de l'echantillon.
    """

    distance_min_px = float("inf")

    for image_index in range(trajectoires_reelles.shape[0]):
        positions = trajectoires_reelles[image_index]

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < distance_min_px:
                    distance_min_px = distance

    if not np.isfinite(distance_min_px):
        return float("inf")

    distance_min_m = (
        distance_min_px * PIXEL_SIZE_M / abs(magnification)
    )

    if tolerance_px >= 0.5 * distance_min_px:
        print(
            "ATTENTION : la tolerance approche la moitie de la distance "
            "minimale entre deux particules. Une mauvaise association est possible."
        )

    if distance_min_m <= 2.0 * radius_m:
        print(
            "ATTENTION : les particules se touchent ou se chevauchent "
            "dans cette simulation. Reduire N_PARTICLES ou augmenter leur separation."
        )

    return distance_min_m


# =============================================================================
# 4. PROGRAMME PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    if len(PARTICLE_INTENSITIES) != N_PARTICLES:
        raise ValueError(
            "PARTICLE_INTENSITIES doit contenir une intensite par particule."
        )


    resultats_par_configuration = {}

    for radius_index in range(len(PARTICLE_RADII_M)):
        radius_m = PARTICLE_RADII_M[radius_index]
        diffusion = calculer_diffusion(radius_m)

        print("\n" + "#" * 75)
        print(f"Rayon simule : {radius_m:.4e} m")
        print(f"Coefficient de diffusion theorique : {diffusion:.4e} m^2/s")

        for config_index in range(len(CONFIGURATIONS)):
            magnification, NA, time_step = CONFIGURATIONS[config_index]

            tolerance_px = tolerance_en_pixels(
                TOLERANCE_M,
                magnification
            )

            sigma_brownien_px = (
                math.sqrt(2.0 * diffusion * time_step)
                * abs(magnification)
                / PIXEL_SIZE_M
            )

            max_deplacement_px = (
                MAX_STEP_SIGMAS * sigma_brownien_px
            )

            demi_depart = calculer_demi_fenetre(
                radius_m,
                magnification,
                NA
            )

            demi_max = max(200, 2 * demi_depart)

            print("\n" + "=" * 70)
            print(
                f"M = {magnification}X, NA = {NA}, "
                f"dt = {time_step} s"
            )
            print(
                f"Tolerance : {TOLERANCE_UM:.3f} um "
                f"= {tolerance_px:.2f} pixels"
            )
            print(
                f"Deplacement brownien RMS par axe : "
                f"{sigma_brownien_px:.2f} pixels"
            )
            print(
                f"Fenetre initiale : demi-largeur = {demi_depart} pixels"
            )

            resultats_sequences = []

            for repetition in range(N_REPETITIONS):
                # Meme graine de base entre objectifs pour une taille
                # donnee : cela facilite une comparaison avec des
                # realisations aleatoires semblables.
                seed = (
                    RANDOM_SEED
                    + 10000 * radius_index
                    + repetition
                )
                rng = np.random.default_rng(seed)

                trajectoires_reelles = generer_trajectoires_pixels(
                    diffusion,
                    time_step,
                    magnification,
                    rng
                )

                distance_min_m = verifier_separation(
                    trajectoires_reelles,
                    tolerance_px,
                    radius_m,
                    magnification
                )

                # Une pile de 10 images float32 occupe environ 0.5 Go.
                # Elle est necessaire parce que suivre_particules()
                # travaille sur une sequence 3D complete.
                data = np.empty(
                    (N_STEPS, NY, NX),
                    dtype=np.float32
                )

                for image_index in range(N_STEPS):
                    image_propre = generer_image_propre(
                        trajectoires_reelles[image_index],
                        magnification,
                        NA,
                        radius_m
                    )

                    image_bruitee = ajouter_bruit_poisson(
                        image_propre,
                        SNR_DB,
                        BACKGROUND_PHOTONS,
                        rng
                    )

                    data[image_index] = image_bruitee

                # Evaluation de l'algorithme complet de suivi.
                traces = suivre_particules(
                    data,
                    demi_depart=demi_depart,
                    k=K_THRESHOLD,
                    aire_min=MIN_AREA,
                    marge=TRACK_MARGIN_PX,
                    demi_max=demi_max,
                    max_deplacement_px=max_deplacement_px,
                    min_fit_snr=MINIMUM_FIT_SNR,
                    marge_bord=demi_depart + TRACK_MARGIN_PX
                )

                mesures = evaluer_suivi(
                    traces,
                    trajectoires_reelles,
                    tolerance_px
                )

                rmse_um = (
                    mesures["RMSE_px"]
                    * PIXEL_SIZE_M
                    / abs(magnification)
                    * 1e6
                )

                faux_positifs_par_image = (
                    mesures["FP"] / N_STEPS
                )

                
                resultats_sequences.append(mesures)

                print(
                    f"Sequence {repetition + 1}/{N_REPETITIONS} : "
                    f"TP={mesures['TP']}, "
                    f"FP={mesures['FP']}, "
                    f"FN={mesures['FN']}, "
                    f"F1={100.0 * mesures['F1']:.1f}%"
                )

                # Liberer la pile d'images avant la repetition suivante.
                del data

            # Calcul de la moyenne et de l'ecart-type entre repetitions.
            precisions = np.array([
                resultat["precision"]
                for resultat in resultats_sequences
            ])

            rappels = np.array([
                resultat["rappel"]
                for resultat in resultats_sequences
            ])

            scores_f1 = np.array([
                resultat["F1"]
                for resultat in resultats_sequences
            ])

            faux_positifs = np.array([
                resultat["FP"] / N_STEPS
                for resultat in resultats_sequences
            ])

            rmse_um_values = np.array([
                resultat["RMSE_px"]
                * PIXEL_SIZE_M
                / abs(magnification)
                * 1e6
                for resultat in resultats_sequences
            ])

            def moyenne_ecart_type(valeurs):
                moyenne = float(np.mean(valeurs))
                if len(valeurs) > 1:
                    ecart_type = float(np.std(valeurs, ddof=1))
                else:
                    ecart_type = 0.0
                return moyenne, ecart_type

            precision_moy, precision_std = moyenne_ecart_type(precisions)
            rappel_moy, rappel_std = moyenne_ecart_type(rappels)
            f1_moy, f1_std = moyenne_ecart_type(scores_f1)
            fp_moy, fp_std = moyenne_ecart_type(faux_positifs)
            rmse_moy, rmse_std = moyenne_ecart_type(rmse_um_values)

            cle = (radius_m, magnification, NA, time_step)
            resultats_par_configuration[cle] = {
                "precision_moy": precision_moy,
                "precision_std": precision_std,
                "rappel_moy": rappel_moy,
                "rappel_std": rappel_std,
                "f1_moy": f1_moy,
                "f1_std": f1_std,
                "fp_moy": fp_moy,
                "fp_std": fp_std,
                "rmse_moy": rmse_moy,
                "rmse_std": rmse_std,
            }

            print("\nBilan moyen entre repetitions :")
            print(
                f"Precision : {100.0 * precision_moy:.2f} "
                f"+/- {100.0 * precision_std:.2f} %"
            )
            print(
                f"Rappel : {100.0 * rappel_moy:.2f} "
                f"+/- {100.0 * rappel_std:.2f} %"
            )
            print(
                f"F1 : {100.0 * f1_moy:.2f} "
                f"+/- {100.0 * f1_std:.2f} %"
            )
            print(
                f"Faux positifs par image : {fp_moy:.2f} "
                f"+/- {fp_std:.2f}"
            )
            print(
                f"RMSE de localisation : {rmse_moy:.3f} "
                f"+/- {rmse_std:.3f} um"
            )

    print("\nSimulation terminee.")
