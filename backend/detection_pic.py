import numpy as np
import tifffile
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, label, maximum_position, sum 
import math

def Load_tif_file_norm(file_path):
    """Retourne le contenu d'un .tif normalisé entre 0 et 1.
    Si le fichier est RGB, on fait la moyenne des 3 couleurs."""
    data = tifffile.imread(file_path)
    val_max = np.iinfo(data.dtype).max       # 255 ou 65535 (255 si 8bits, 65535 si 16bits)
    data = data / val_max                    # normalise selon le nombre de bit
    if data.ndim == 4:                       # si RGB : (pages, hauteur, largeur, 3)
        data = data.mean(axis=-1)            # on fait la moyenne des 3 couleurs
    return data, val_max


#####################################################################################################
# Moyenne des positions des différents pixels pondérée par leur intensité
#####################################################################################################

def centroide(image):
    """Retourne le centroïde d'une image 2D (x0, y0) en pixels. C'est à dire 
    la moyenne des positions des différents pixels pondérée par leur intensité."""
    image = (image - np.median(image)).clip(min=0)   # on enlève le fond
    y, x = np.indices(image.shape)                   # grilles de positions
    total = image.sum()
    return (x * image).sum() / total, (y * image).sum() / total

def all_tif_pos_cent(chemin):
    """Retourne les positions x0 et y0 pour toutes les pages d'un .tif"""
    data, val_max = Load_tif_file_norm(chemin)
    positions = []
    for i in range(data.shape[0]):
        x0, y0 = centroide(data[i])
        positions.append([x0, y0])
    return positions

def position_centroid(data):
    """Retourne les positions x0 et y0 pour toutes les pages d'un numpy array 3D (images 2D)"""
    positions = []
    for i in range(data.shape[0]):
        x0, y0 = centroide(data[i])
        positions.append([x0, y0])
    return positions


#####################################################################################################
# Fit gaussien en 1D
#####################################################################################################

def gauss(x, a, x0, sigma, b):
    # Gaussian function
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

def fit_profil(profil):
    """Fit gaussien d'un profil 1D. Retourne les paramètres et leurs incertitudes."""
    profil = np.asarray(profil, dtype=float)

    # Les premiers if sont pour détecter les erreurs de profil avant de lancer le fit, pour éviter des messages d'erreur incompréhensibles.
    if profil.ndim != 1 or len(profil) < 3:
        raise ValueError("Le profil doit etre un tableau 1D d'au moins 3 points.")

    if not np.all(np.isfinite(profil)):
        raise ValueError("Le profil contient des valeurs non finies.")

    fond_initial = float(np.median(profil))
    amplitude_initiale = float(np.max(profil) - fond_initial)

    if amplitude_initiale <= 0.0:
        raise RuntimeError("Aucun pic positif dans le profil.")

    # Paramètres initiaux pour le fit gaussien
    centre_initial = float(np.argmax(profil))
    sigma_initial = max(1.0, len(profil) / 10.0)
    print("sigma inital: ", sigma_initial)

    p0 = [
        amplitude_initiale,
        centre_initial,
        sigma_initial,
        fond_initial
    ]

    x = np.arange(len(profil), dtype=float)

    bornes_inferieures = [0.0, 0.0, 0.3, -np.inf]
    bornes_superieures = [np.inf, len(profil) - 1.0, len(profil), np.inf]

    popt, pcov = curve_fit(
        gauss,
        x,
        profil,
        p0=p0,
        bounds=(bornes_inferieures, bornes_superieures),
        max_nfev=10000
    )

    erreurs = np.sqrt(np.diag(pcov))
    return popt, erreurs

def gaussian_fit(image):
    """image : UNE page (tableau 2D). Retourne x0, y0 en pixels et leurs incertitudes.
    On fait la somme des lignes et des colonnes pour obtenir les profils selon x et y en 1D,
    puis on fit un gaussien sur ces profils."""

    profil_x = image.sum(axis=0)   # on additionne les lignes -> profil selon x
    profil_y = image.sum(axis=1)   # on additionne les colonnes -> profil selon y

    popt_x, err_x = fit_profil(profil_x)
    popt_y, err_y = fit_profil(profil_y)

    # On prend le sigma le plus grand pour garder une fenêtre assez grande pour englober la particule. 
    sigma = max(popt_x[2], popt_y[2])

    return popt_x[1], popt_y[1], err_x[1], err_y[1], sigma

def erreur_localisation(image, sigma, val_max):
    N = ((image - np.median(image)).clip(min=0).sum())*val_max
    return sigma / np.sqrt(N)

def all_tif_pos_gauss(chemin):
    """Retourne les positions x0, y0 et leurs incertitudes pour toutes les pages d'un .tif"""
    data, val_max = Load_tif_file_norm(chemin)
    positions = []
    for i in range(data.shape[0]):
        x0, y0, err_x, err_y, sigma = gaussian_fit(data[i])
        err_loc = erreur_localisation(data[i], sigma, val_max)
        positions.append([x0, y0, err_x, err_y, err_loc])
    return positions

def position_gaussian(data):
    """Retourne les positions x0, y0 et leurs incertitudes pour toutes les pages d'un numpy array 3D (images 2D)
    input : data (numpy array 3D)
    output : positions (list of lists) : [[x0, y0, err_x, err_y, err_loc], ...]
    """
    positions = []
    for i in range(data.shape[0]):
        x0, y0, err_x, err_y, sigma = gaussian_fit(data[i])
        err_loc = erreur_localisation(data[i], sigma, 1)
        positions.append([x0, y0, err_x, err_y, err_loc])
    return positions

#####################################################################################################
# Suivit de plusieurs particule avec fit gaussien
#####################################################################################################

def decouper(image, x, y, demi=15):
    """Découpe un petit encadré de l'image autour du point (x, y).

    Cette fonction retourne trois choses :
      - fenetre : le morceau d'image découpé, d'environ (2*demi+1) x (2*demi+1) pixels
      - x1, y1 : la position du coin en haut à gauche de l'encadré, dans l'image complète

    Pour revenir à la position dans l'image complète, à partir d'une position
    (x0, y0) trouvée dans l'encadré (par un fit, par exemple) :
        x_image = x0 + x1
        y_image = y0 + y1
    """
    x, y = int(round(x)), int(round(y))
    x1, x2 = max(x - demi, 0), min(x + demi + 1, image.shape[1])
    y1, y2 = max(y - demi, 0), min(y + demi + 1, image.shape[0])
    return image[y1:y2, x1:x2], x1, y1

def trouver_particules(image, k=5, aire_min=5, marge_bord=0):
    """Retourne la liste des positions (x, y) des particules trouvées."""

    image = np.asarray(image, dtype=float)

    lisse = gaussian_filter(image, 2)
    fond = np.median(lisse)

    # Pour convertir Median Absolute Deviation (MAD) en écart-type, on multiplie par 1.4826
    mad = np.median(np.abs(lisse - fond))
    bruit = 1.4826 * mad

    # Seul les pixels qui dépasse le seuil sont considérés
    # k signifie au moins x fois la taille du bruit au-dessus du fond
    seuil = fond + k * bruit
    masque = lisse > seuil


    # Les pixels qui dépasse le seuil sont numéroté                   
    etiquettes, nombre_regions = label(masque)                       
    if nombre_regions == 0:
        return []
    
    indices = range(1, nombre_regions + 1)

    aires = sum(masque, etiquettes, indices)
    pics = maximum_position(lisse, etiquettes, indices)

    #On garde seulement les taches assez grandes
    positions = []
    for region_index in range(nombre_regions):
        aire = aires[region_index]
        if aire < aire_min:
            continue
        ligne, colonne = pics[region_index]

        # Une tache coupee par le bord ne peut pas etre ajustee correctement.
        if colonne < marge_bord or colonne >= image.shape[1] - marge_bord:
            continue

        if ligne < marge_bord or ligne >= image.shape[0] - marge_bord:
            continue
        
        positions.append((colonne, ligne))   # (x, y) = (colonne, ligne)
    return positions


def suivre_particules(data, demi_depart=20, k=5, aire_min=5, marge=10, demi_max=200, max_deplacement_px=None, marge_bord=0):
    """data : tableau (pages, hauteur, largeur). Retourne une liste de positions par particule."""
    positions_initiales = trouver_particules(data[0], k, aire_min, marge_bord)
    derniere_position_valide = [tuple(position) for position in positions_initiales] # On garde la dernière position valide pour chaque particule
    demi_fenetres = [int(demi_depart) for _ in positions_initiales] # On garde la taille de la fenêtre pour chaque particule
    echecs_consecutifs = [0 for _ in positions_initiales] # On garde le nombre d'échecs consécutifs pour chaque particule
    
    traces = [[] for _ in positions_initiales]

    for image_index, image in enumerate(data):                          # toutes les images, la première incluse
        for track_index in range(len(traces)):

            x_prec, y_prec = derniere_position_valide[track_index]
            demi = demi_fenetres[track_index] # On garde la taille de la fenêtre pour chaque particule

            fenetre, x1, y1 = decouper(image, x_prec, y_prec, demi)
            try:
                x0, y0, err_x, err_y, sigma = gaussian_fit(fenetre)
                x = x0 + x1
                y = y0 + y1                         # retour aux coordonnées de l'image

                valeurs_valides = np.all(np.isfinite([x, y, err_x, err_y, sigma]))

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