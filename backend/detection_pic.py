import numpy as np
import tifffile
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, label, maximum_position, sum 

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
    x = np.arange(len(profil))

    # Valeurs de départ 
    p0 = [profil.max() - profil.min(), np.argmax(profil), 3, profil.min()]

    popt, pcov = curve_fit(gauss, x, profil, p0=p0)
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
    return popt_x[1], popt_y[1], err_x[1], err_y[1], popt_x[2]

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
    """Retourne l'encadré autour de (x, y) et son coin en haut à gauche."""
    x, y = int(round(x)), int(round(y))
    x1, x2 = max(x - demi, 0), min(x + demi + 1, image.shape[1])
    y1, y2 = max(y - demi, 0), min(y + demi + 1, image.shape[0])
    return image[y1:y2, x1:x2], x1, y1

def trouver_particules(image, k=5, aire_min=5):
    """Retourne la liste des positions (x, y) des particules trouvées."""
    lisse = gaussian_filter(image, 2)                   # réduit le bruit
    fond = np.median(lisse)
    bruit = 1.4826 * np.median(np.abs(lisse - fond))    # estimation du bruit
    masque = lisse > fond + k * bruit                   # seuil
    etiquettes, n = label(masque)                       # numérote chaque tache
    if n == 0:
        return []
    indices = range(1, n + 1)
    aires = sum(masque, etiquettes, indices)
    pics = maximum_position(lisse, etiquettes, indices)
    return [(x, y) for (y, x), aire in zip(pics, aires) if aire >= aire_min]

def suivre_particules(data, demi=20):
    """data : tableau (pages, hauteur, largeur). Retourne une trace par particule."""
    positions = trouver_particules(data[0])          # toutes les particules de la 1re image
    traces = [[p] for p in positions]                # une liste de positions par particule
    derniere = list(positions)                       # dernière position valide de chacune

    for image in data[1:]:
        for k, trace in enumerate(traces):
            x_prec, y_prec = derniere[k]
            fenetre, x1, y1 = decouper(image, x_prec, y_prec, demi)
            try:
                x0, y0, err_x, err_y, sigma = gaussian_fit(fenetre)
                x, y = x0 + x1, y0 + y1              # retour aux coordonnées de l'image
                trace.append((x, y))
                derniere[k] = (x, y)
            except RuntimeError:                     # fit raté : on garde la trace, on note nan
                trace.append((np.nan, np.nan))
    return traces