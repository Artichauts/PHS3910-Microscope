import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
import os
from backend.detection_pic import suivre_particules

# Chemin du chemin initial de l'ordinateur jusqu'au fichier .tif
# Film_D_0.1_exp_0.5s_PixelSize_0.5um_Mono_8bit.tif
# Film_D_0.1_exp_0.5s_PixelSize_0.5um_Mono_16bit.tif
# Film_D_0.1_exp_0.5s_PixelSize_0.5um_RGB_8bitt.tif
chemin = Path(__file__).parent / "captures" / "output_frames.npz"
# Remplacer 'donnees.npz' par le chemin de votre fichier


# Charger le fichier NPZ
with np.load(chemin) as data:
    print("Contenu du fichier :", data.files)
    images = data["images"]

    suivis = suivre_particules(
        images,
        demi_depart=20,
        k=5,
        aire_min=5,
        marge=10,
        demi_max=200,
        max_deplacement_px=None,
        marge_bord=0
    )


dossier = "images_detectees"

# Enregistrer chaque image
for i, image in enumerate(images):

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image, cmap="gray")

    for particule in suivis:
        if i < len(particule):
            x, y = particule[i]

            if np.isfinite(x) and np.isfinite(y):
                ax.scatter(x, y, color="red", s=40, marker="x")

    ax.set_title(f"Image {i+1} - Détection des particules")
    ax.axis("off")

    fig.savefig(
        os.path.join(dossier, f"image_{i:05d}.png"),
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

print(f"Images enregistrées dans le dossier : {dossier}")


