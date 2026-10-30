from backend.brown_movement import generate_brownian_motion
from backend.fct_PSF import particle_image
import numpy as np

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

    Lambda = 532e-9  # longueur d'onde (m)
    R = 2e-6 

    # Paramètres pour l'image de particule
    NA = 0.9
    M = -40
    pixel_size = 1.55e-6
    R = 2e-6

    image_array = np.zeros((N_steps, max(x), max(y)))  # Tableau pour stocker les images générées
    # Générer l'image de la particule à partir des positions générées
    for i in range(N_steps):
        img = particle_image(NA=0.9, M=-40, pixel_size=1.55e-6, radius=R, LAMBDA=Lambda,
                        shape=(max(x), max(y)), center=(x[i], y[i]))
        # Ici vous pouvez ajouter du code pour visualiser ou sauvegarder l'image générée si nécessaire.