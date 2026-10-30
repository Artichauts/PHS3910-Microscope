from backend.brown_movement import generate_brownian_motion
from backend.fct_PSF import particle_image
from backend.detection_pic import position_gaussian, position_centroid

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Paramètres pour le mouvement brownien
    N_steps = 100  # Nombre de pas de trajectoire
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

    dim_x = int(np.ceil(np.max(x))) + 1
    dim_y = int(np.ceil(np.max(y))) + 1


    image_array = np.zeros((N_steps, dim_x, dim_y))  # Tableau pour stocker les images générées
    
    # Générer l'image de la particule à partir des positions générées

    for i in range(N_steps):
        img = particle_image(NA=0.9, M=-40, pixel_size=1.55e-6, radius=R, LAMBDA=Lambda,
                        shape=(dim_x, dim_y), center=(x[i], y[i]))
        image_array[i] = img
        # Ici vous pouvez ajouter du code pour visualiser ou sauvegarder l'image générée si nécessaire.

    positions_gaussian = position_gaussian(image_array)
    position_centroid = position_centroid(image_array)

    plt.plot(x, y, label='Trajectoire Brownienne')
    plt.scatter(positions_gaussian[:, 0], positions_gaussian[:, 1], color='red', label='Positions Gaussiennes')
    plt.scatter(position_centroid[:, 0], position_centroid[:, 1], color='green', label='Positions Centroides')
    plt.xlabel('Position X (A.U)')
    plt.ylabel('Position Y (A.U)')
    plt.title('Localisation des particules')
    plt.legend()
    plt.show()
    