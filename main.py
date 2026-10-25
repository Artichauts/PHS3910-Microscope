import numpy as np
import tifffile
from pathlib import Path
from backend.detection_pic import Load_tif_file_norm, gaussian_fit

# Chemin du chemin initial de l'ordinateur jusqu'au fichier .tif
chemin = Path(__file__).parent / "sim_prof" / "Film_D_0.1_exp_0.5s_PixelSize_0.5um_Mono_8bit.tif"


data = Load_tif_file_norm(chemin)
print(data.shape)
print(data)