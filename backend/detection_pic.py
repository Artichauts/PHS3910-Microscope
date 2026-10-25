import numpy as np
import tifffile
from scipy.optimize import curve_fit


def Load_tif_file_norm(file_path):
    """Retourne le contenu d'un .tif normalisé entre 0 et 1.
    Si le fichier est RGB, on fait la moyenne des 3 couleurs."""
    data = tifffile.imread(file_path)
    data = data / np.iinfo(data.dtype).max   # normalise selon le nombre de bit (255 si 8bits, 65535 si 16bits)
    if data.ndim == 4:                       # si RGB : (pages, hauteur, largeur, 3)
        data = data.mean(axis=-1)            # on fait la moyenne des 3 couleurs
    return data

def gauss(x, a, x0, sigma, b):
    # Gaussian function
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

def fit_profil(profil):
    """Fit gaussien d'un profil 1D. Retourne les paramètres et leurs incertitudes."""
    x = np.arange(len(profil))
    # Valeurs de départ : le fit en a besoin pour bien converger
    p0 = [profil.max() - profil.min(), np.argmax(profil), 2, profil.min()]
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
    return popt_x[1], popt_y[1], err_x[1], err_y[1]

