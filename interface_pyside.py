"""Interface de départ pour le projet de suivi de particules.

À ADAPTER :
    - Les imports ci-dessous doivent correspondre aux noms réels de vos fichiers.
    - capture_video doit accepter num_frames, filename, fps, exposure_us,
      magnification, scale_um_px et bit_depth, et créer le fichier JSON décrit
      dans les explications qui accompagnent ce fichier.
    - Le FPS utilisé pour l'analyse doit être le vrai FPS d'acquisition, pas
      seulement le FPS inscrit dans le conteneur MP4.

Dépendances côté interface/analyse :
    pip install pyside6 opencv-python numpy

Sur le Raspberry Pi, Picamera2 est généralement installé avec apt :
    sudo apt install python3-picamera2 python3-opencv
"""

from __future__ import annotations

import sys
import traceback
import json
from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# -----------------------------------------------------------------------------
# À ADAPTER AUX NOMS DE VOS FICHIERS PYTHON
# -----------------------------------------------------------------------------
from camera import capture_video
from detection_pic import suivre_particules
from analyse_msd import msd_fft, regression


K_B = 1.380649e-23  # Constante de Boltzmann, J/K


# -----------------------------------------------------------------------------
# LECTURE VIDÉO IMAGE PAR IMAGE : aucune séquence 3D complète en RAM
# -----------------------------------------------------------------------------
def lire_video_images(chemin: str | Path):
    """Générateur qui donne une image en niveaux de gris à la fois."""
    chemin = Path(chemin)
    hauteur_brute = None
    chemin_metadata = chemin.with_suffix(".json")
    if chemin_metadata.exists():
        try:
            with chemin_metadata.open("r", encoding="utf-8") as fichier:
                metadata = json.load(fichier)
            hauteur_brute = metadata.get("raw_height")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            hauteur_brute = None

    cap = cv2.VideoCapture(str(chemin))

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {chemin}")

    try:
        while True:
            succes, image_bgr = cap.read()
            if not succes:
                break

            # Si la vidéo contient une bande de métadonnées sous l'image,
            # on l'enlève avant la détection pour ne pas analyser le texte.
            if hauteur_brute is not None:
                image_bgr = image_bgr[:int(hauteur_brute), :]
            image_grise = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            yield image_grise
    finally:
        cap.release()


# -----------------------------------------------------------------------------
# OUTILS POUR L'ANALYSE DU RAYON
# -----------------------------------------------------------------------------
def plus_long_segment_valide(trace: list[tuple[float, float]]) -> np.ndarray:
    """Retourne le plus long segment consécutif sans NaN dans une trajectoire."""
    points = np.asarray(trace, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return np.empty((0, 2), dtype=float)

    valide = np.all(np.isfinite(points), axis=1)
    changements = np.diff(np.r_[False, valide, False].astype(int))
    debuts = np.where(changements == 1)[0]
    fins = np.where(changements == -1)[0]

    if len(debuts) == 0:
        return np.empty((0, 2), dtype=float)

    debut, fin = max(zip(debuts, fins), key=lambda intervalle: intervalle[1] - intervalle[0])
    return points[debut:fin]


def analyser_video(
    chemin_video: str,
    fps_acquisition: float,
    echelle_um_px: float,
    temperature_c: float,
    viscosite_pa_s: float,
    max_deplacement_px: float,
    lag_max: int,
    timestamps_s: list[float] | None = None,
) -> dict[str, Any]:
    """Suit les particules et estime un rayon pour chaque trajectoire exploitable.

    Hypothèses de cette première version :
      - l'acquisition est à fréquence constante;
      - le mouvement est analysé en 2D;
      - les positions sont converties en mètres avant de calculer le MSD;
      - la pente du MSD vaut 4D en 2D.

    L'incertitude calculée à la fin est la dispersion statistique entre les
    rayons obtenus. Elle n'inclut pas encore les incertitudes de calibration,
    température, viscosité ou ajustement de chaque pente.
    """
    if fps_acquisition <= 0:
        raise ValueError("Le FPS d'acquisition doit être supérieur à zéro.")
    if echelle_um_px <= 0:
        raise ValueError("L'échelle en µm/pixel doit être supérieure à zéro.")
    if viscosite_pa_s <= 0:
        raise ValueError("La viscosité doit être supérieure à zéro.")

    # suivre_particules doit accepter un itérable/générateur d'images.
    traces = suivre_particules(
        lire_video_images(chemin_video),
        max_deplacement_px=max_deplacement_px,
    )

    # msd_fft suppose des intervalles de temps constants. Si un fichier de
    # métadonnées fournit les temps de capture, on estime le pas réel et on
    # refuse une séquence trop irrégulière plutôt que de produire un rayon trompeur.
    if timestamps_s is not None and len(timestamps_s) >= 2:
        intervalles = np.diff(np.asarray(timestamps_s, dtype=float))
        if np.any(intervalles <= 0):
            raise ValueError("Les horodatages ne sont pas strictement croissants.")
        dt = float(np.median(intervalles))
        irregularite = float(np.std(intervalles) / np.mean(intervalles))
        if irregularite > 0.05:
            raise ValueError(
                f"La cadence est trop irrégulière (CV={irregularite:.1%}). "
                "msd_fft suppose un échantillonnage régulier."
            )
        fps_acquisition = 1.0 / dt

    temperature_k = temperature_c + 273.15
    metres_par_pixel = echelle_um_px * 1e-6
    rayons_m = []
    incertitudes_rayons_m = []
    details = []

    for numero, trace in enumerate(traces, start=1):
        segment = plus_long_segment_valide(trace)

        # Il faut assez de points pour ajuster plusieurs retards du MSD.
        if len(segment) < 5:
            continue

        positions_m = segment * metres_par_pixel
        msd = np.asarray(msd_fft(positions_m), dtype=float)

        nombre_lags = min(int(lag_max), len(msd) - 1)
        if nombre_lags < 3:
            continue

        temps = np.arange(1, nombre_lags + 1, dtype=float) / fps_acquisition
        msd_fit = msd[1:nombre_lags + 1]

        if not np.all(np.isfinite(msd_fit)):
            continue

        # a_x=0 et a_y=0 demandent à votre fonction regression de calculer
        # l'incertitude de pente à partir des résidus du fit seulement.
        # Cela reste une estimation indicative : les points du MSD sont corrélés.
        modele, pente, intercept, incertitude_pente, incertitude_intercept, r2 = regression(
            temps, msd_fit, a_x=np.zeros_like(temps), a_y=0.0
        )

        if not np.isfinite(pente) or pente <= 0:
            continue

        # En 2D : MSD = 4 D t, donc pente = 4D.
        diffusion = pente / 4.0
        rayon_m = K_B * temperature_k / (6.0 * np.pi * viscosite_pa_s * diffusion)
        incertitude_rayon_m = (
            rayon_m * abs(incertitude_pente / pente)
            if np.isfinite(incertitude_pente) else float("nan")
        )

        if np.isfinite(rayon_m) and rayon_m > 0:
            rayons_m.append(rayon_m)
            incertitudes_rayons_m.append(incertitude_rayon_m)
            details.append({
                "track": numero,
                "rayon_um": rayon_m * 1e6,
                "incertitude_fit_um": incertitude_rayon_m * 1e6,
                "r2": float(r2),
                "nb_points": int(len(segment)),
            })

    if not rayons_m:
        raise RuntimeError(
            "Aucune trajectoire assez longue n'a permis d'estimer un rayon. "
            "Vérifie la détection, le suivi, le FPS et les paramètres physiques."
        )

    rayons_um = np.asarray(rayons_m) * 1e6
    moyenne_um = float(np.mean(rayons_um))

    if len(rayons_um) >= 2:
        ecart_type_um = float(np.std(rayons_um, ddof=1))
        incertitude_interparticulaire_um = ecart_type_um / np.sqrt(len(rayons_um))
        incertitudes_fit_um = np.asarray(incertitudes_rayons_m, dtype=float) * 1e6
        incertitudes_fit_um = incertitudes_fit_um[np.isfinite(incertitudes_fit_um)]
        incertitude_fit_moyenne_um = (
            float(np.sqrt(np.sum(incertitudes_fit_um ** 2)) / len(rayons_um))
            if len(incertitudes_fit_um) == len(rayons_um) else float("nan")
        )
        incertitude_moyenne_um = (
            float(np.hypot(incertitude_interparticulaire_um, incertitude_fit_moyenne_um))
            if np.isfinite(incertitude_fit_moyenne_um)
            else incertitude_interparticulaire_um
        )
    else:
        ecart_type_um = float("nan")
        incertitude_interparticulaire_um = float("nan")
        incertitude_fit_moyenne_um = float("nan")
        incertitude_moyenne_um = float("nan")

    return {
        "nombre": len(rayons_um),
        "moyenne_um": moyenne_um,
        "ecart_type_um": ecart_type_um,
        "incertitude_interparticulaire_um": incertitude_interparticulaire_um,
        "incertitude_fit_moyenne_um": incertitude_fit_moyenne_um,
        "incertitude_moyenne_um": incertitude_moyenne_um,
        "details": details,
        "traces": traces,
    }


# -----------------------------------------------------------------------------
# EXÉCUTION DES TÂCHES LOURDES SANS BLOQUER L'INTERFACE
# -----------------------------------------------------------------------------
class Travailleur(QThread):
    termine = Signal(object)
    erreur = Signal(str)

    def __init__(self, fonction: Callable, *args, **kwargs):
        super().__init__()
        self.fonction = fonction
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            resultat = self.fonction(*self.args, **self.kwargs)
            self.termine.emit(resultat)
        except Exception:
            self.erreur.emit(traceback.format_exc())


# -----------------------------------------------------------------------------
# FENÊTRE PRINCIPALE
# -----------------------------------------------------------------------------
class FenetrePrincipale(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suivi de particules browniennes")
        self.resize(1150, 760)

        self.chemin_video: str | None = None
        self.capture_video = None
        self.index_image = 0
        self.nombre_images = 0
        self.timestamps_s: list[float] | None = None
        self.metadata: dict[str, Any] = {}
        self.worker: Travailleur | None = None

        self._construire_interface()

    def _construire_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QHBoxLayout(central)

        # ---------------- Panneau de gauche : paramètres et commandes ----------------
        panneau = QVBoxLayout()
        layout_principal.addLayout(panneau, 0)

        groupe_acquisition = QGroupBox("Acquisition")
        formulaire_acquisition = QFormLayout(groupe_acquisition)

        self.nb_images = QSpinBox()
        self.nb_images.setRange(1, 100000)
        self.nb_images.setValue(100)

        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.01, 240.0)
        self.fps.setDecimals(2)
        self.fps.setValue(0.9)

        self.exposition = QSpinBox()
        self.exposition.setRange(1, 1000000)
        self.exposition.setValue(1000)
        self.exposition.setSuffix(" µs")

        self.grossissement = QComboBox()
        self.grossissement.addItems(["4x", "10x", "20x", "40x", "60x"])
        self.grossissement.setCurrentText("10x")

        self.echelle = QDoubleSpinBox()
        self.echelle.setRange(0.0001, 10000.0)
        self.echelle.setDecimals(5)
        self.echelle.setValue(0.25)  # À remplacer par votre calibration réelle.
        self.echelle.setSuffix(" µm/px")

        self.profondeur = QSpinBox()
        self.profondeur.setRange(1, 16)
        self.profondeur.setValue(8)
        self.profondeur.setSuffix(" bits")

        formulaire_acquisition.addRow("Nombre d'images", self.nb_images)
        formulaire_acquisition.addRow("Fréquence cible", self.fps)
        formulaire_acquisition.addRow("Exposition", self.exposition)
        formulaire_acquisition.addRow("Grossissement", self.grossissement)
        formulaire_acquisition.addRow("Échelle spatiale", self.echelle)
        formulaire_acquisition.addRow("Profondeur", self.profondeur)
        panneau.addWidget(groupe_acquisition)

        groupe_analyse = QGroupBox("Analyse physique")
        formulaire_analyse = QFormLayout(groupe_analyse)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(-20.0, 100.0)
        self.temperature.setDecimals(2)
        self.temperature.setValue(20.0)
        self.temperature.setSuffix(" °C")

        self.viscosite = QDoubleSpinBox()
        self.viscosite.setRange(1e-8, 10.0)
        self.viscosite.setDecimals(6)
        self.viscosite.setValue(0.001002)  # Valeur indicative, à confirmer pour l'échantillon.
        self.viscosite.setSuffix(" Pa·s")

        self.deplacement = QSpinBox()
        self.deplacement.setRange(1, 10000)
        self.deplacement.setValue(15)
        self.deplacement.setSuffix(" px")

        self.lag_max = QSpinBox()
        self.lag_max.setRange(3, 10000)
        self.lag_max.setValue(20)

        formulaire_analyse.addRow("Température", self.temperature)
        formulaire_analyse.addRow("Viscosité", self.viscosite)
        formulaire_analyse.addRow("Déplacement maximal", self.deplacement)
        formulaire_analyse.addRow("Nombre maximal de retards", self.lag_max)
        panneau.addWidget(groupe_analyse)

        self.bouton_acquerir = QPushButton("Acquérir une séquence")
        self.bouton_acquerir.clicked.connect(self.acquerir_sequence)
        panneau.addWidget(self.bouton_acquerir)

        self.bouton_ouvrir = QPushButton("Ouvrir une vidéo")
        self.bouton_ouvrir.clicked.connect(self.ouvrir_video)
        panneau.addWidget(self.bouton_ouvrir)

        self.bouton_analyser = QPushButton("Analyser la vidéo")
        self.bouton_analyser.clicked.connect(self.lancer_analyse)
        panneau.addWidget(self.bouton_analyser)

        panneau.addStretch(1)

        # ---------------- Panneau de droite : image et navigation ----------------
        affichage = QVBoxLayout()
        layout_principal.addLayout(affichage, 1)

        self.image_label = QLabel("Ouvre une vidéo ou acquiers une séquence.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #202020; color: white;")
        affichage.addWidget(self.image_label, 1)

        navigation = QHBoxLayout()
        self.bouton_precedent = QPushButton("◀ Image précédente")
        self.bouton_precedent.clicked.connect(lambda: self.afficher_image(self.index_image - 1))
        self.bouton_suivant = QPushButton("Image suivante ▶")
        self.bouton_suivant.clicked.connect(lambda: self.afficher_image(self.index_image + 1))
        navigation.addWidget(self.bouton_precedent)
        navigation.addWidget(self.bouton_suivant)
        affichage.addLayout(navigation)

        self.info_image = QLabel("Aucune vidéo ouverte")
        affichage.addWidget(self.info_image)

        self.resultat = QLabel("Résultat de l'analyse : —")
        self.resultat.setWordWrap(True)
        affichage.addWidget(self.resultat)

        self.statut = QLabel("Prêt")
        affichage.addWidget(self.statut)

    # ------------------------------------------------------------------
    # Affichage d'une image du fichier vidéo
    # ------------------------------------------------------------------
    def ouvrir_video(self):
        chemin, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir une vidéo",
            "",
            "Vidéos (*.mp4 *.avi *.mov *.mkv);;Tous les fichiers (*)",
        )
        if chemin:
            self.charger_video(chemin)

    def charger_video(self, chemin: str):
        if self.capture_video is not None:
            self.capture_video.release()

        cap = cv2.VideoCapture(chemin)
        if not cap.isOpened():
            QMessageBox.critical(self, "Erreur", f"Impossible d'ouvrir :\n{chemin}")
            return

        self.capture_video = cap
        self.chemin_video = chemin
        self.timestamps_s = None
        self.metadata = {}
        chemin_metadata = Path(chemin).with_suffix(".json")
        if chemin_metadata.exists():
            try:
                with chemin_metadata.open("r", encoding="utf-8") as fichier:
                    self.metadata = json.load(fichier)
                valeurs = self.metadata.get("timestamps_s")
                if isinstance(valeurs, list):
                    self.timestamps_s = [float(v) for v in valeurs]
                if "fps_target" in self.metadata:
                    self.fps.setValue(float(self.metadata["fps_target"]))
                if "scale_um_px" in self.metadata:
                    self.echelle.setValue(float(self.metadata["scale_um_px"]))
                if "exposure_us" in self.metadata:
                    self.exposition.setValue(int(self.metadata["exposure_us"]))
                if "magnification" in self.metadata:
                    self.grossissement.setCurrentText(str(self.metadata["magnification"]))
                if "bit_depth" in self.metadata:
                    self.profondeur.setValue(int(self.metadata["bit_depth"]))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.metadata = {}
                self.timestamps_s = None

        self.nombre_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index_image = 0
        self.afficher_image(0)
        self.statut.setText(f"Vidéo ouverte : {Path(chemin).name}")

    def afficher_image(self, index: int):
        if self.capture_video is None or self.nombre_images <= 0:
            return

        index = max(0, min(index, self.nombre_images - 1))
        self.capture_video.set(cv2.CAP_PROP_POS_FRAMES, index)
        succes, image_bgr = self.capture_video.read()

        if not succes:
            self.statut.setText("Impossible de lire cette image.")
            return

        self.index_image = index
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Les métadonnées sont affichées dans l'interface.
        # Le temps index/FPS est exact seulement si l'acquisition est régulière.
        fps = self.fps.value()
        temps = (
            self.timestamps_s[index]
            if self.timestamps_s is not None and index < len(self.timestamps_s)
            else (index / fps if fps > 0 else 0.0)
        )
        if not self.metadata.get("overlay_in_video", False):
            texte = (
                f"t = {temps:.3f} s | {self.echelle.value():.5f} µm/px | "
                f"{self.grossissement.currentText()} | "
                f"expo. {self.exposition.value()} µs | "
                f"{fps:g} images/s | {self.profondeur.value()} bits"
            )
            cv2.putText(
                image_rgb,
                texte,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        hauteur, largeur, canaux = image_rgb.shape
        qimage = QImage(
            image_rgb.data,
            largeur,
            hauteur,
            canaux * largeur,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self.info_image.setText(
            f"Image {index + 1}/{self.nombre_images} — "
            f"temps affiché : {temps:.3f} s"
        )

    # ------------------------------------------------------------------
    # Acquisition : capture_video est exécutée dans un autre thread.
    # ------------------------------------------------------------------
    def acquerir_sequence(self):
        chemin, _ = QFileDialog.getSaveFileName(
            self,
            "Enregistrer la séquence",
            "sequence.mp4",
            "Vidéo MP4 (*.mp4)",
        )
        if not chemin:
            return
        if not chemin.lower().endswith(".mp4"):
            chemin += ".mp4"

        arguments = {
            "num_frames": self.nb_images.value(),
            "filename": chemin,
            "fps": self.fps.value(),
            "exposure_us": self.exposition.value(),
            "magnification": self.grossissement.currentText(),
            "scale_um_px": self.echelle.value(),
            "bit_depth": self.profondeur.value(),
        }

        self._activer_commandes(False)
        self.statut.setText("Acquisition en cours...")
        self.worker = Travailleur(self._fonction_acquisition, arguments)
        self.worker.termine.connect(self._acquisition_terminee)
        self.worker.erreur.connect(self._erreur_travailleur)
        self.worker.start()

    @staticmethod
    def _fonction_acquisition(arguments: dict[str, Any]):
        # Cette signature suppose que vous avez ajouté exposure_us à capture_video.
        capture_video(**arguments)
        return arguments["filename"]

    def _acquisition_terminee(self, chemin: str):
        self._activer_commandes(True)
        self.charger_video(chemin)
        self.statut.setText("Acquisition terminée.")

    # ------------------------------------------------------------------
    # Analyse : suivi + MSD + rayon
    # ------------------------------------------------------------------
    def lancer_analyse(self):
        if not self.chemin_video:
            QMessageBox.information(self, "Aucune vidéo", "Ouvre d'abord une vidéo.")
            return

        parametres = {
            "chemin_video": self.chemin_video,
            "fps_acquisition": self.fps.value(),
            "echelle_um_px": self.echelle.value(),
            "temperature_c": self.temperature.value(),
            "viscosite_pa_s": self.viscosite.value(),
            "max_deplacement_px": self.deplacement.value(),
            "lag_max": self.lag_max.value(),
            "timestamps_s": self.timestamps_s,
        }

        self._activer_commandes(False)
        self.statut.setText("Détection, suivi et calcul du rayon en cours...")
        self.worker = Travailleur(analyser_video, **parametres)
        self.worker.termine.connect(self._analyse_terminee)
        self.worker.erreur.connect(self._erreur_travailleur)
        self.worker.start()

    def _analyse_terminee(self, resultat: dict[str, Any]):
        self._activer_commandes(True)
        n = resultat["nombre"]
        moyenne = resultat["moyenne_um"]
        ecart_type = resultat["ecart_type_um"]
        incertitude = resultat["incertitude_moyenne_um"]

        if n >= 2:
            message = (
                f"Rayon moyen : {moyenne:.4g} µm\n"
                f"Écart-type entre particules : {ecart_type:.3g} µm\n"
                f"Incertitude indicative sur la moyenne : {incertitude:.3g} µm\n"
                f"Nombre de trajectoires utilisées : {n}"
            )
        else:
            message = (
                f"Rayon estimé : {moyenne:.4g} µm\n"
                "Une seule trajectoire exploitable : impossible d'estimer la dispersion entre particules."
            )

        self.resultat.setText(message)
        self.statut.setText("Analyse terminée.")
        print("Détails des rayons :")
        for detail in resultat["details"]:
            print(detail)

    def _erreur_travailleur(self, message: str):
        self._activer_commandes(True)
        self.statut.setText("Une erreur est survenue.")
        QMessageBox.critical(self, "Erreur", message)

    def _activer_commandes(self, actif: bool):
        self.bouton_acquerir.setEnabled(actif)
        self.bouton_ouvrir.setEnabled(actif)
        self.bouton_analyser.setEnabled(actif)

    def closeEvent(self, event):
        if self.capture_video is not None:
            self.capture_video.release()
            self.capture_video = None

        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()

        event.accept()


def main():
    app = QApplication(sys.argv)
    fenetre = FenetrePrincipale()
    fenetre.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
