import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

N_steps = 100 # Nombre de pas de trajectoire

D = 1 #Coefficient de diffusion (um^2/s)

time_step = 0.1 # Longueur de chaque image (s)

delta_positions = np.random.normal(loc=0, scale= math.sqrt(2*D*time_step),

size=(N_steps,3))

positions_xy = np.cumsum(delta_positions, axis=0)
x, y, z = positions_xy.T
timestamp =  np.linspace(0, N_steps*time_step, N_steps)


plt.plot(x,y)
plt.show()


def cumstd(positions):
  std = pd.DataFrame(positions).expanding().std(ddof=0).to_numpy()
  return std


  
def regression(x, y, a_x= None, a_y= None):
    """
    Regression lineaire
    """
    N = len(x)
    print("Lenght of the sample ", N)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x*y)
    sum_x2 = np.sum(x**2)
    sum_x_xm = np.sum((x-np.mean(x))**2)

    Delta = N*sum_x_xm
    print(f"Delta : {Delta}")
    #Delta = N*sum_x2 - (sum_x)**2

    # formule p.11
    m =  (N*sum_xy - sum_x*sum_y)/Delta
    print("m factor : ", m)
    c = (sum_x2*sum_y - sum_x*sum_xy)/Delta
    def myfunc(x):
        return m * x + c

    mymodel = list(map(myfunc, x))

    r2 = 1 - np.sum((y-mymodel)**2)/np.sum((y-np.mean(y))**2)

    if a_x is not None and a_y is not None:

      a_x_y = np.mean(m*a_x)

      sigma_y = np.sqrt(np.sum((y-m*x-c)**2)/(N-2))

      print(f"sigma_y : {sigma_y}")
      print(f"U_b : {a_y}")
      print(f"a_x_y:{a_x_y}")


      # On ajoute en quadrature l'incertitude en x
      a_y = np.sqrt((a_x_y)**2 + (a_y)**2 + (sigma_y)**2)
      print(f"a_y : {a_y}")

      #print(f"U_c**2 + U_b**2 : {a_y}")
      a_m = a_y*np.sqrt(N/Delta)
      a_c = a_y*np.sqrt(sum_x2/Delta)
      print(f"a_m : {a_m}, a_c : {a_c}")



      return mymodel, m, c, a_m, a_c, r2
    else :
      return mymodel, m, c, r2

    