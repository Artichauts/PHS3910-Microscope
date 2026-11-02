import math
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from scipy.ndimage import maximum_position
from scipy.ndimage import sum as ndi_sum
from scipy.optimize import curve_fit


# =============================================================================
# 1. AJUSTEMENT GAUSSIEN 1D ET 2D SEPARE
# =============================================================================

def gauss(x, amplitude, centre, sigma, fond):
    """Modele gaussien 1D avec un fond constant."""
    return amplitude * np.exp(-((x - centre) ** 2) / (2.0 * sigma ** 2)) + fond


def fit_profil(profil):
    """Ajuste une gaussienne a un profil 1D.

    Retourne :
        parametres : amplitude, centre, sigma et fond
        erreurs : incertitudes estimees par curve_fit

    Les bornes empechent notamment un sigma negatif ou un centre en dehors
    du profil. curve_fit etant deja utilise dans le projet, aucune nouvelle
    dependance n'est necessaire.
    """

    profil = np.asarray(profil, dtype=float)

    if profil.ndim != 1 or len(profil) < 3:
        raise ValueError("Le profil doit etre un tableau 1D d'au moins 3 points.")

    if not np.all(np.isfinite(profil)):
        raise ValueError("Le profil contient des valeurs non finies.")

    x = np.arange(len(profil), dtype=float)

    fond_initial = float(np.median(profil))
    amplitude_initiale = float(np.max(profil) - fond_initial)

    if amplitude_initiale <= 0.0:
        raise RuntimeError("Aucun pic positif dans le profil.")

    centre_initial = float(np.argmax(profil))
    sigma_initial = max(1.0, len(profil) / 10.0)

    p0 = [
        amplitude_initiale,
        centre_initial,
        sigma_initial,
        fond_initial
    ]

    bornes_inferieures = [0.0, 0.0, 0.3, -np.inf]
    bornes_superieures = [np.inf, len(profil) - 1.0, len(profil), np.inf]

    parametres, covariance = curve_fit(
        gauss,
        x,
        profil,
        p0=p0,
        bounds=(bornes_inferieures, bornes_superieures),
        maxfev=10000
    )

    erreurs = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return parametres, erreurs


def gaussian_fit(image):
    """Localise une tache dans une image 2D avec deux fits gaussiens 1D.

    On somme les lignes et les colonnes pour obtenir les profils x et y.
    Le centre obtenu est exprime dans les coordonnees de la petite image.

    Retour : x0, y0, erreur_x, erreur_y, sigma_moyen
    """

    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("gaussian_fit attend une image 2D.")

    if min(image.shape) < 3:
        raise ValueError("La fenetre est trop petite pour un fit gaussien.")

    if not np.all(np.isfinite(image)):
        raise ValueError("L'image contient des valeurs non finies.")

    profil_x = np.sum(image, axis=0)
    profil_y = np.sum(image, axis=1)

    parametres_x, erreurs_x = fit_profil(profil_x)
    parametres_y, erreurs_y = fit_profil(profil_y)

    x0 = float(parametres_x[1])
    y0 = float(parametres_y[1])

    erreur_x = float(erreurs_x[1])
    erreur_y = float(erreurs_y[1])

    sigma_x = float(parametres_x[2])
    sigma_y = float(parametres_y[2])
    sigma_moyen = 0.5 * (sigma_x + sigma_y)

    return x0, y0, erreur_x, erreur_y, sigma_moyen


# =============================================================================
# 2. DETECTION DES PARTICULES DANS UNE IMAGE
# =============================================================================

def trouver_particules(
    image,
    k=5,
    aire_min=5,
    sigma_lissage=2.0,
    marge_bord=0
):
    """Retourne les positions (x, y) des regions lumineuses detectees.

    Le seuil est : mediane + k * bruit robuste.
    Le bruit robuste est estime a partir du MAD :
        bruit = 1.4826 * mediane(abs(image - mediane(image)))

    Les composantes sont connectees avec les 8 voisins. Cela evite qu'une
    tache diagonale soit artificiellement separee en plusieurs regions.
    Les pics trop proches du bord peuvent etre ignores avec marge_bord.

    Attention : une seule composante connexe donne une seule detection.
    Deux particules dont les regions lumineuses se touchent peuvent donc
    etre fusionnees en une detection.
    """

    image = np.asarray(image, dtype=float)

    if image.ndim != 2:
        raise ValueError("trouver_particules attend une image 2D.")

    if not np.all(np.isfinite(image)):
        raise ValueError("L'image contient des valeurs non finies.")

    if k <= 0:
        raise ValueError("k doit etre strictement positif.")

    if aire_min < 1:
        raise ValueError("aire_min doit etre au moins egale a 1.")

    # Le lissage reduit les fluctuations pixel a pixel avant le seuillage.
    image_lissee = gaussian_filter(image, sigma=sigma_lissage)

    fond = float(np.median(image_lissee))
    mad = float(np.median(np.abs(image_lissee - fond)))
    bruit = 1.4826 * mad

    # Si le MAD vaut zero (par exemple avec une image entierement quantifiee),
    # on utilise une estimation minimale du bruit de comptage photonique.
    if bruit <= np.finfo(float).eps:
        bruit = math.sqrt(max(fond, 1.0))

    seuil = fond + k * bruit
    masque = image_lissee > seuil

    # Structure 3x3 : les pixels horizontaux, verticaux et diagonaux
    # sont consideres comme voisins.
    structure = np.ones((3, 3), dtype=np.uint8)
    etiquettes, nombre_regions = label(masque, structure=structure)

    if nombre_regions == 0:
        return []

    indices = np.arange(1, nombre_regions + 1)

    aires = ndi_sum(
        masque.astype(np.uint8),
        labels=etiquettes,
        index=indices
    )

    pics = maximum_position(
        image_lissee,
        labels=etiquettes,
        index=indices
    )

    positions = []

    for region_index in range(nombre_regions):
        aire = int(aires[region_index])

        if aire < aire_min:
            continue

        ligne, colonne = pics[region_index]

        # Une tache coupee par le bord ne peut pas etre ajustee correctement.
        if colonne < marge_bord or colonne >= image.shape[1] - marge_bord:
            continue

        if ligne < marge_bord or ligne >= image.shape[0] - marge_bord:
            continue

        # Retourner les coordonnees dans l'ordre (x, y) = (colonne, ligne).
        positions.append((float(colonne), float(ligne)))

    return positions


# =============================================================================
# 3. DECOUPAGE D'UNE FENETRE AUTOUR D'UNE POSITION
# =============================================================================

def decouper(image, x, y, demi=15):
    """Decoupe une fenetre autour de (x, y).

    Retourne la fenetre et les coordonnees (x1, y1) de son coin superieur
    gauche dans l'image complete.
    """

    x = int(round(x))
    y = int(round(y))
    demi = int(max(1, demi))

    x1 = max(x - demi, 0)
    x2 = min(x + demi + 1, image.shape[1])

    y1 = max(y - demi, 0)
    y2 = min(y + demi + 1, image.shape[0])

    fenetre = image[y1:y2, x1:x2]

    return fenetre, x1, y1


# =============================================================================
# 4. SUIVI DES PARTICULES SUR UNE SEQUENCE
# =============================================================================

def suivre_particules(
    data,
    demi_depart=20,
    k=5,
    aire_min=5,
    marge=10,
    demi_max=200,
    max_deplacement_px=None,
    min_fit_snr=1.5,
    marge_bord=0
):
    """Detecte les particules dans la premiere image puis les suit.

    Parametres
    ----------
    data : ndarray 3D
        Sequence de forme (nombre_images, hauteur, largeur).
    demi_depart : int
        Demi-largeur initiale de la fenetre de fit, en pixels.
    k : float
        Seuil de detection de la premiere image.
    aire_min : int
        Aire minimale d'une region detectee.
    marge : int
        Marge ajoutee a la fenetre lorsque le fit reussit ou echoue.
    demi_max : int
        Demi-largeur maximale d'une fenetre.
    max_deplacement_px : float ou None
        Deplacement maximal accepte entre deux images, en pixels.
        Si None, aucun filtre de deplacement n'est applique.
    min_fit_snr : float
        Contraste minimal du pic par rapport au bruit estime dans la fenetre.
        Cette valeur filtre certains fits effectues uniquement sur du bruit.
    marge_bord : int
        Distance minimale au bord de l'image pour accepter une detection
        initiale. Utile pour eviter les taches tronquees par le bord.

    Retour
    ------
    traces : liste de trajectoires
        Chaque trajectoire contient une paire (x, y) par image.
        Un echec de localisation est represente par (nan, nan).

    La fonction ne cree pas de nouvelle trajectoire apres la premiere image.
    Une particule non detectee dans la premiere image ne sera donc pas suivie.
    """

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError("data doit avoir la forme (images, hauteur, largeur).")

    if data.shape[0] < 1:
        raise ValueError("La sequence doit contenir au moins une image.")

    # Etape 1 : detection de tous les candidats dans la premiere image.
    positions_initiales = trouver_particules(
        data[0],
        k=k,
        aire_min=aire_min,
        marge_bord=marge_bord
    )

    derniere_position_valide = [tuple(position) for position in positions_initiales]
    demi_fenetres = [int(demi_depart) for _ in positions_initiales]
    echecs_consecutifs = [0 for _ in positions_initiales]

    traces = [[] for _ in positions_initiales]

    # Etape 2 : suivre chaque candidat dans chaque image.
    for image_index in range(data.shape[0]):
        image = data[image_index]

        for track_index in range(len(traces)):

            x_prec, y_prec = derniere_position_valide[track_index]
            demi = demi_fenetres[track_index]

            fenetre, x1, y1 = decouper(
                image,
                x_prec,
                y_prec,
                demi
            )

            try:
                # Verifier qu'il existe bien un signal local dans la fenetre.
                # Le bruit est estime par MAD, puis compare au pic de la fenetre.
                fond_local = float(np.median(fenetre))
                mad_local = float(
                    np.median(np.abs(fenetre - fond_local))
                )
                bruit_local = 1.4826 * mad_local

                if bruit_local <= np.finfo(float).eps:
                    bruit_local = math.sqrt(max(fond_local, 1.0))

                contraste_local = float(np.max(fenetre) - fond_local)

                if contraste_local < min_fit_snr * bruit_local:
                    raise RuntimeError("Signal trop faible dans la fenetre.")

                x0, y0, erreur_x, erreur_y, sigma = gaussian_fit(fenetre)

                x = x0 + x1
                y = y0 + y1

                valeurs_valides = np.all(
                    np.isfinite([x, y, erreur_x, erreur_y, sigma])
                )

                if not valeurs_valides:
                    raise RuntimeError("Le fit contient des valeurs non finies.")

                if sigma <= 0.0:
                    raise RuntimeError("La largeur gaussienne est invalide.")

                # Le fit doit rester dans la fenetre extraite.
                if x0 < 0.0 or x0 >= fenetre.shape[1]:
                    raise RuntimeError("Le centre x sort de la fenetre.")

                if y0 < 0.0 or y0 >= fenetre.shape[0]:
                    raise RuntimeError("Le centre y sort de la fenetre.")

                # Eviter qu'un fit saute vers une autre tache tres eloignee.
                # Apres un echec, la limite augmente avec le nombre d'images
                # manquees afin de permettre une eventuelle reacquisition.
                if max_deplacement_px is not None and image_index > 0:
                    distance = math.sqrt(
                        (x - x_prec) ** 2 + (y - y_prec) ** 2
                    )

                    limite = max_deplacement_px * (
                        echecs_consecutifs[track_index] + 1
                    )

                    if distance > limite:
                        raise RuntimeError("Deplacement trop important pour une image.")

                # Le fit est accepte : enregistrer et mettre a jour la position.
                traces[track_index].append((x, y))
                derniere_position_valide[track_index] = (x, y)
                echecs_consecutifs[track_index] = 0

                # Adapter la fenetre a la largeur estimee de la tache.
                nouveau_demi = int(math.ceil(5.0 * abs(sigma))) + marge
                nouveau_demi = max(int(demi_depart), nouveau_demi)
                nouveau_demi = min(nouveau_demi, int(demi_max))
                demi_fenetres[track_index] = nouveau_demi

            except (RuntimeError, ValueError, FloatingPointError, OverflowError):
                # Ne pas remplacer la derniere position valide par un NaN.
                traces[track_index].append((np.nan, np.nan))
                echecs_consecutifs[track_index] += 1

                # Agrandir progressivement la fenetre pour tenter de retrouver
                # la particule dans l'image suivante.
                demi_fenetres[track_index] = min(
                    demi + marge,
                    int(demi_max)
                )

    return traces
