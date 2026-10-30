import numpy as np
from scipy.special import j1
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

LAMBDA = 532e-9  # longueur d'onde (m)

def psf(NA, r, M):
    """
    PSF d'Airy normalisée (intensité) dans le plan image d'un système 4f.

    Paramètres
    ----------
    NA : float
        Ouverture numérique côté objet (celle de l'objectif).
    r : float ou ndarray
        Distance radiale dans le plan image (détecteur), en mètres.
    M : float
        Grandissement du système, M = -f2/f1 (le signe est ignoré).

    Retourne
    --------
    float ou ndarray : PSF(r), avec PSF(0) = 1.
    """
    r = np.asarray(r, dtype=float)
    x = 2 * np.pi * NA * r / (abs(M) * LAMBDA)

    # Limite en x -> 0 : 2*J1(x)/x -> 1, donc PSF -> 1
    ratio = np.divide(2 * j1(x), x, out=np.ones_like(x), where=x != 0)

    return ratio**2

def psf_image(NA, M, pixel_size, shape=(101, 101), center=None, oversample=1):
    """
    Simule l'image d'un émetteur ponctuel sur une caméra.

    Paramètres
    ----------
    NA : float
        Ouverture numérique côté objet.
    M : float
        Grandissement du système (le signe est ignoré).
    pixel_size : float
        Taille d'un pixel de la caméra, en mètres (ex. 6.5e-6).
    shape : (int, int)
        Taille de l'image (ny, nx), en pixels.
    center : (float, float) ou None
        Position (x0, y0) de l'émetteur, en pixels. Les centres de pixels sont
        aux valeurs entières. Par défaut : centre de l'image. Les valeurs
        non entières permettent de simuler un émetteur entre deux pixels.
    oversample : int
        Sous-échantillonnage par pixel : la PSF est calculée sur une grille
        plus fine puis moyennée sur chaque pixel (intégration sur la surface
        du pixel, comme sur un vrai capteur). 1 = valeur au centre du pixel.

    Retourne
    --------
    ndarray (ny, nx) : image de la PSF, avec un maximum de 1 si l'émetteur
    est centré sur un pixel.
    """
    ny, nx = shape
    x0, y0 = center if center is not None else ((nx - 1) / 2, (ny - 1) / 2)
    s = int(oversample)

    # coordonnées (en pixels) des sous-échantillons ; le pixel i couvre [i-0.5, i+0.5]
    xs = (np.arange(nx * s) + 0.5) / s - 0.5
    ys = (np.arange(ny * s) + 0.5) / s - 0.5
    X, Y = np.meshgrid(xs, ys)

    r = np.hypot(X - x0, Y - y0) * pixel_size
    img = psf(NA, r, M)

    # moyenne sur les s x s sous-échantillons de chaque pixel
    return img.reshape(ny, s, nx, s).mean(axis=(1, 3))

def particle_image(NA, M, pixel_size, radius, shape=(101, 101), center=None,
                   profile="sphere", oversample=5, kernel_airy_radii=10):
    """
    Simule l'image d'une particule ronde émissive sur une caméra.

    Paramètres
    ----------
    NA : float
        Ouverture numérique côté objet.
    M : float
        Grandissement du système (le signe est ignoré).
    pixel_size : float
        Taille d'un pixel de la caméra, en mètres.
    radius : float
        Rayon réel de la particule (dans l'échantillon), en mètres
        (ex. 50e-9 à 10e-6).
    shape : (int, int)
        Taille de l'image (ny, nx), en pixels.
    center : (float, float) ou None
        Position (x0, y0) de la particule, en pixels (centres de pixels aux
        entiers). Par défaut : centre de l'image.
    profile : "disk" ou "sphere"
        Profil d'émission de la particule.
    oversample : int
        Sous-échantillonnage par pixel (intégration sur la surface du pixel).
    kernel_airy_radii : float
        Demi-largeur du noyau de PSF, en rayons d'Airy (la PSF a de longues
        queues ; 10 suffit en général, ~98 % de l'énergie).

    Retourne
    --------
    ndarray (ny, nx) : fraction des photons émis reçue par chaque pixel
    (somme ≈ 1 si la particule est entièrement dans le champ).
    Multiplier par le nombre total de photons pour avoir des comptes.
    """
    ny, nx = shape
    x0, y0 = center if center is not None else ((nx - 1) / 2, (ny - 1) / 2)
    s = int(oversample)
    d = pixel_size / s                      # pas de la grille fine (m)

    # --- noyau de PSF sur la grille fine ---
    r_airy = 0.61 * abs(M) * LAMBDA / NA    # rayon d'Airy dans le plan image
    hk = int(np.ceil(kernel_airy_radii * r_airy / d))
    k = np.arange(-hk, hk + 1)
    KX, KY = np.meshgrid(k, k)
    kernel = psf(NA, np.hypot(KX, KY) * d, M)
    kernel /= kernel.sum()

    # --- objet sur la grille fine, avec marge hk pour éviter les effets de bord ---
    xs = (np.arange(nx * s + 2 * hk) - hk + 0.5) / s - 0.5
    ys = (np.arange(ny * s + 2 * hk) - hk + 0.5) / s - 0.5
    X, Y = np.meshgrid(xs, ys)
    rho = np.hypot(X - x0, Y - y0) * pixel_size   # distance au centre (m, plan image)
    R_img = abs(M) * radius                       # rayon de la particule dans le plan image

    if profile == "disk":
        obj = (rho <= R_img).astype(float)
    elif profile == "sphere":
        obj = np.sqrt(np.clip(R_img**2 - rho**2, 0, None))
    else:
        raise ValueError("profile doit être 'disk' ou 'sphere'")

    if obj.sum() == 0:                            # particule plus petite que le pas de grille
        obj[np.unravel_index(rho.argmin(), rho.shape)] = 1.0   # -> émetteur ponctuel
    obj /= obj.sum()

    # --- convolution, puis intégration sur la surface de chaque pixel ---
    fine = fftconvolve(obj, kernel, mode="valid")   # taille (ny*s, nx*s)
    return fine.reshape(ny, s, nx, s).sum(axis=(1, 3))


R = 2e-6 
img = particle_image(NA=0.9, M=-40, pixel_size=1.55e-6, radius=R,
                     shape=(380, 507), center=(201.7, 289.1))

plt.imshow(img, cmap="gray", origin="lower")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.show()

# oversample: Un vrai pixel de caméra n’échantillonne pas la PSF en un seul point : il intègre la lumière sur toute sa surface. Avec oversample=1, on calcule la PSF seulement au centre de chaque pixel (approximation ponctuelle). Avec oversample=s, on découpe chaque pixel en une grille de $s \times s$ sous-points, on calcule la PSF en chacun, puis on fait la moyenne. Plus s est grand, plus c’est fidèle à un vrai capteur, mais plus le calcul est long. Quand le pic de la PSF est étalé sur plusieurs pixels, oversample=1 suffit presque. Quand il ne couvre que 1 à 3 pixels, oversample=3 à 5 fait une vraie différence.