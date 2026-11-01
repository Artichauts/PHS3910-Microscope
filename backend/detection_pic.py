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

def trouver_particules(image, k=5, aire_min=5):
    """Retourne la liste des positions (x, y) des particules trouvées."""

    lisse = gaussian_filter(image, 2)
    fond = np.median(lisse)

    # Pour convertir Median Absolute Deviation (MAD) en écart-type, on multiplie par 1.4826
    bruit = 1.4826 * np.median(np.abs(lisse - fond)) 

    # Seul les pixels qui dépasse le seuil sont considérés
    # k signifie au moins x fois la taille du bruit au-dessus du fond
    seuil = fond + k * bruit
    masque = lisse > seuil

    print("masque", masque)
    # Les pixels qui dépasse le seuil sont numéroté                   
    etiquettes, n = label(masque)                       
    if n == 0:
        return []
    
    indices = range(1, n + 1)

    aires = sum(masque, etiquettes, indices)
    pics = maximum_position(lisse, etiquettes, indices)

    #On garde seulement les taches assez grandes
    positions = []
    for i in range(n):
        aire = aires[i]
        ligne, colonne = pics[i]
        if aire >= aire_min:
            positions.append((colonne, ligne))   # (x, y) = (colonne, ligne)
    return positions


def suivre_particules(data, demi_depart=20, k=5, aire_min=5, marge=10, demi_max=200):
    """data : tableau (pages, hauteur, largeur). Retourne une liste de positions par particule."""
    positions = trouver_particules(data[0], k, aire_min)
    derniere = list(positions)                  # dernière position valide de chaque particule
    demis = [demi_depart] * len(positions)      # demi-largeur de l'encadré de chaque particule
    traces = [[] for _ in positions]            # positions de chaque particule, image par image

    for image in data:                          # toutes les images, la première incluse
        for j in range(len(traces)):
            x_prec, y_prec = derniere[j]
            fenetre, x1, y1 = decouper(image, x_prec, y_prec, demis[j])
            try:
                x0, y0, err_x, err_y, sigma = gaussian_fit(fenetre)
                x, y = x0 + x1, y0 + y1                         # retour aux coordonnées de l'image
                traces[j].append((x, y))
                derniere[j] = (x, y)
                demis[j] = min(int(5 * abs(sigma)) + marge, demi_max)   # l'encadré s'adapte
            except RuntimeError:                                # le fit a échoué
                traces[j].append((np.nan, np.nan))
    return traces