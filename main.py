import numpy as np
import tifffile
from pathlib import Path
from backend.detection_pic import Load_tif_file_norm, gaussian_fit, all_tif_pos

# Chemin du chemin initial de l'ordinateur jusqu'au fichier .tif
chemin = Path(__file__).parent / "sim_prof" / "Film_D_0.1_exp_0.5s_PixelSize_0.5um_Mono_8bit.tif"


# data = Load_tif_file_norm(chemin)
# x0, y0, err_x, err_y = gaussian_fit(data[0])   # première page seulement
# print(f"x:{x0:.2f} ± {err_x:.2f} & y:{y0:.2f} ± {err_y:.2f} ")
data = all_tif_pos(chemin)
print(data)
