from backend.brown_movement import generate_brownian_motion
from backend.fct_PSF import particle_image
from backend.detection_pic import position_gaussian, position_centroid
from backend.regression import regression, msd_fft

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def analyse_one_molecule_image(image_array, pixel_size, M):
    """
    Analyse une séquence d'images pour localiser les particules et calculer le MSD.
    Le input est une sequence d'images (tableau 3D) et les paramètres de la caméra et du système optique.
    L'image ne doit contenir qu'une seule particule.


    Paramètres
    ----------
    image_array : ndarray (N_steps, ny, nx)
        Tableau contenant les images générées.
    pixel_size : float
        Taille d'un pixel de la caméra, en mètres.
    M : float
        Grandissement du système (le signe est ignoré).

    Retourne
    --------
    positions_gaussian_array : ndarray (N_steps, 2)
        Positions x, y des particules localisées par fit gaussien.
    position_centroid_array : ndarray (N_steps, 2)
        Positions x, y des particules localisées par centroïde.
    msd_gaussian : ndarray (N_steps,)
        MSD calculé à partir des positions gaussiennes.
    msd_centroid : ndarray (N_steps,)
        MSD calculé à partir des positions centroïdes.
    model_gaussian : ndarray (N_steps,)
        Modèle de régression linéaire pour le MSD gaussien.
    model_centroid : ndarray (N_steps,)
        Modèle de régression linéaire pour le MSD centroïde.
    m_gaussian : float
        Pente de la régression linéaire pour le MSD gaussien.
    m_centroid : float
        Pente de la régression linéaire pour le MSD centroïde.

    """
    positions_gaussian_array = position_gaussian(image_array)
    position_centroid_array = position_centroid(image_array)

    Mabs = abs(M)
    positions_gaussian_array = np.array(positions_gaussian_array) * pixel_size / Mabs
    position_centroid_array  = np.array(position_centroid_array)  * pixel_size / Mabs


    msd_gaussian = msd_fft(positions_gaussian_array)
    msd_centroid = msd_fft(position_centroid_array )

    model_gaussian, m_gaussian, c_gaussian, r2_gaussian = regression(timestamp, msd_gaussian)
    model_centroid, m_centroid, c_centroid, r2_centroid = regression(timestamp, msd_centroid)


    return positions_gaussian_array, position_centroid_array, msd_gaussian, msd_centroid, model_gaussian, model_centroid, m_gaussian, m_centroid


if __name__ == "__main__":

    # Paramètres pour le mouvement brownien
    N_steps = 50  # Nombre de pas de trajectoire
    R = 2e-6   # rayon de la particule (m)
    kb = 1.380649e-23  # constante de Boltzmann (J/K)
    T = 298  # température (K)
    nu = 1.01e-3 # viscosité de l'eau (Pa.s)
    D = kb * T / (6 * np.pi * nu * R)  # Coefficient de diffusion (m^2/s)
    print(f"Coefficient de diffusion théorique : {D:.4e} m^2/s")


    #D = 1          # Coefficient de diffusion (um^2/s)
    time_step = 0.05  # Longueur de chaque image (s)

    # Générer la trajectoire de mouvement brownien
    positions_xy, timestamp = generate_brownian_motion(N_steps, D, time_step)
    print("Positions XY, shape and type:\n", positions_xy.shape, type(positions_xy))

    x = positions_xy[:, 0]
    y = positions_xy[:, 1]

    print("Position max and min :")
    print("X : ", np.max(x), " - ", np.min(x))
    print("Y : ", np.max(y), " - ", np.min(y))

    Lambda = 575e-9  # longueur d' (m)

    # Paramètres pour l'image de particule
    NA = 0.45
    M = -10
    pixel_size = 1.55e-6

    dim_x = np.int64(np.max(x) - np.min(x)) + 1
    dim_y = np.int64(np.max(y) - np.min(y)) + 1

    ny, nx = 507, 380

    image_array = np.zeros((N_steps, ny, nx))  # Tableau pour stocker les images générées

    # Générer l'image de la particule à partir des positions générées
    px_size = 1.55e-6  # taille du pixel (m)
    Mabs = abs(M)
    for i in tqdm(range(N_steps), desc="Génération des images"):
        img = particle_image(NA=0.45, M=-10, pixel_size=px_size, radius=R, LAMBDA=Lambda,
                        shape=(ny, nx), center=(x[i]*Mabs/px_size + nx/2, y[i]*Mabs/px_size + ny/2))
        image_array[i] = img
        # Ici vous pouvez ajouter du code pour visualiser ou sauvegarder l'image générée si nécessaire.

    plt.imshow(image_array[0], cmap="gray", origin="lower")
    plt.show()


    positions_gaussian_array = position_gaussian(image_array)
    position_centroid_array = position_centroid(image_array)

    positions_gaussian_array = np.array(positions_gaussian_array) * pixel_size / Mabs
    position_centroid_array  = np.array(position_centroid_array)  * pixel_size / Mabs

    plt.scatter(positions_gaussian_array[:, 0], positions_gaussian_array[:, 1],
                color='red', label='Positions Gaussiennes')


    plt.scatter(position_centroid_array[:, 0], position_centroid_array[:, 1], color='green', label='Positions Centroides')
    
    plt.plot(x + nx/2*pixel_size/Mabs, y + ny/2*pixel_size/Mabs, label='Trajectoire Brownienne')

    plt.xlabel('Position X (A.U)')
    plt.ylabel('Position Y (A.U)')
    plt.title('Localisation des particules')
    plt.legend()
    plt.show()


    msd_gaussian = msd_fft(positions_gaussian_array)
    msd_centroid = msd_fft(position_centroid_array )

    model_gaussian, m_gaussian, c_gaussian, r2_gaussian = regression(timestamp, msd_gaussian)
    model_centroid, m_centroid, c_centroid, r2_centroid = regression(timestamp, msd_centroid)

    plt.plot(timestamp, msd_gaussian, label='MSD Gaussien')
    plt.plot(timestamp, msd_centroid, label='MSD Centroid')
    plt.plot(timestamp, model_gaussian, label='Régression Gaussienne')
    plt.plot(timestamp, model_centroid, label='Régression Centroid')
    plt.plot(timestamp, 4 * D * time_step * np.arange(N_steps), label='Théorie')
    plt.xlabel('Nombre de pas')
    plt.ylabel('MSD (A.U)')
    plt.title('MSD cumulatif des positions')
    plt.legend()
    plt.show()

    D_gauss = m_gaussian / 4
    D_centroid = m_centroid / 4

    R_gauss = kb * T / (6 * np.pi * nu * D_gauss)
    R_centroid = kb * T / (6 * np.pi * nu * D_centroid)

    print(f"Diffusion Gaussienne : {D_gauss:.4e} m^2/s, Rayon estimé : {R_gauss:.4e} m, erreur : {np.abs(R_gauss - R)/R:.4%}")
    print(f"Diffusion Centroid : {D_centroid:.4e} m^2/s, Rayon estimé : {R_centroid:.4e} m, erreur : {np.abs(R_centroid - R)/R:.4%}")