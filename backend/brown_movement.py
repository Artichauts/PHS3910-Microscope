import numpy as np
import math
import matplotlib.pyplot as plt



def generate_brownian_motion(N_steps=100, D=1, time_step=0.1):
    """
    Génère une trajectoire de mouvement brownien en 3D.

    Paramètres
    ----------
    N_steps : int
        Nombre de pas de la trajectoire.
    D : float
        Coefficient de diffusion (um^2/s).
    time_step : float
        Longueur de chaque image (s).

    Retourne
    --------
    positions_xy : ndarray (N_steps, 3)
        Positions x, y, z à chaque pas.
    timestamp : ndarray (N_steps,)
        Temps correspondant à chaque pas.
    """
    delta_positions = np.random.normal(loc=0, scale=math.sqrt(2 * D * time_step), size=(N_steps, 2))
    positions_xy = np.cumsum(delta_positions, axis=0)
    timestamp = np.linspace(0, N_steps * time_step, N_steps)
    
    return positions_xy, timestamp


if __name__ == "__main__":
    N_steps = 100  # Nombre de pas de trajectoire
    D = 1          # Coefficient de diffusion (um^2/s)
    time_step = 0.1  # Longueur de chaque image (s)

    positions_xy, timestamp = generate_brownian_motion(N_steps, D, time_step)
    print("Positions XY, shape and type:\n", positions_xy.shape, type(positions_xy))
    x = positions_xy[:, 0]
    y = positions_xy[:, 1]

    plt.plot(x, y)
    plt.xlabel('Position X (A.U)')
    plt.ylabel('Position Y (A.U)')
    plt.title('Trajectoire de Mouvement Brownien en 2D')
    plt.axis('equal')
    plt.grid()
    plt.show()
