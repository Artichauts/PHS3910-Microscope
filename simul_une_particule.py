from backend.brown_movement import generate_brownian_motion
from backend.fct_PSF import particle_image
from backend.detection_pic import position_gaussian, position_centroid

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    # Paramètres pour le mouvement brownien
    N_steps = 3  # Nombre de pas de trajectoire
    D = 1          # Coefficient de diffusion (um^2/s)
    time_step = 0.1  # Longueur de chaque image (s)

    # Générer la trajectoire de mouvement brownien
    positions_xy, timestamp = generate_brownian_motion(N_steps, D, time_step)
    print("Positions XY, shape and type:\n", positions_xy.shape, type(positions_xy))

    x = positions_xy[:, 0]
    y = positions_xy[:, 1]

    print("Position max and min :")
    print("X : ", np.max(x), " - ", np.min(x))
    print("Y : ", np.max(y), " - ", np.min(y))

    Lambda = 532e-9  # longueur d'onde (m)
    R = 2e-6 

    # Paramètres pour l'image de particule
    NA = 0.9
    M = -40
    pixel_size = 1.55e-6
    R = 2e-6

    dim_x = np.int64(np.max(x) - np.min(x)) + 1
    dim_y = np.int64(np.max(y) - np.min(y)) + 1


    image_array = np.zeros((N_steps, 380, 507))  # Tableau pour stocker les images générées

    # Générer l'image de la particule à partir des positions générées

    for i in tqdm(range(N_steps), desc="Génération des images"):
        img = particle_image(NA=0.9, M=-40, pixel_size=1.55e-6, radius=R, LAMBDA=Lambda,
                        shape=(380, 507), center=(x[i] + 380/2, y[i] + 507/2))
        image_array[i] = img
        # Ici vous pouvez ajouter du code pour visualiser ou sauvegarder l'image générée si nécessaire.

    plt.imshow(image_array[0], cmap="gray", origin="lower")
    plt.show()


    positions_gaussian = position_gaussian(image_array)
    position_centroid = position_centroid(image_array)


    positions_gaussian = np.array(positions_gaussian)

    plt.scatter(positions_gaussian[:, 0], positions_gaussian[:, 1],
                color='red', label='Positions Gaussiennes')
    position_centroid = np.array(position_centroid)

    plt.scatter(position_centroid[:, 0], position_centroid[:, 1], color='green', label='Positions Centroides')
    
    plt.plot(x + 380/2, y + 507/2, label='Trajectoire Brownienne')

    plt.xlabel('Position X (A.U)')
    plt.ylabel('Position Y (A.U)')
    plt.title('Localisation des particules')
    plt.legend()
    plt.show()
    