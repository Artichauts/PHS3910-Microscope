import numpy as np
import tifffile
from pathlib import Path
from backend.detection_pic import all_tif_pos_cent, all_tif_pos_gauss

# Chemin du chemin initial de l'ordinateur jusqu'au fichier .tif
# Film_D_0.1_exp_0.5s_PixelSize_0.5um_Mono_8bit.tif
# Film_D_0.1_exp_0.5s_PixelSize_0.5um_Mono_16bit.tif
# Film_D_0.1_exp_0.5s_PixelSize_0.5um_RGB_8bitt.tif
chemin = Path(__file__).parent / "sim_prof" / "Film_D_0.1_exp_0.5s_PixelSize_0.5um_Mono_16bit.tif"


data = all_tif_pos_gauss(chemin)
data2 = all_tif_pos_cent(chemin)
print(data)
print(data2)
