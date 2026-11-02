"""Acquisition Picamera2 vers un fichier .npy memory-mapped + visualiseur PyQt6.

- Pas de prévisualisation pendant l'acquisition.
- Les images sont écrites une par une dans un .npy avec open_memmap.
- Les pixels enregistrés sont le plan Y (luminance) du flux YUV420 : uint8, 1 canal.
- Les horodatages du capteur sont enregistrés dans un second .npy.
- Un fichier JSON contient le nombre réel d'images écrites et les paramètres.

À exécuter sur le Raspberry Pi avec Picamera2 installé.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Acquisition dans un thread séparé
# ---------------------------------------------------------------------------

class AcquisitionThread(QThread):
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    completed = pyqtSignal(str, int, str)
    failed = pyqtSignal(str)

    def __init__(self, filename: str, n_frames: int, fps: float,
                 exposure_us: int, width: int, height: int):
        super().__init__()
        self.filename = Path(filename)
        self.n_frames = int(n_frames)
        self.fps = float(fps)
        self.exposure_us = int(exposure_us)
        self.requested_width = int(width)
        self.requested_height = int(height)
        self._stop_event = threading.Event()

    def request_stop(self):
        """Demande un arrêt propre après l'image en cours."""
        self._stop_event.set()

    def run(self):
        camera = None
        images_map = None
        times_map = None
        frames_written = 0
        error_message = ""
        width = height = None
        first_timestamp_ns = None
        timestamp_source = "SensorTimestamp Picamera2"
        start_wall = time.perf_counter()
        flush_seconds = 0.0
        first_capture_metadata = {}

        timestamps_path = self.filename.with_name(
            self.filename.stem + "_timestamps.npy"
        )
        metadata_path = self.filename.with_suffix(".json")

        try:
            # Import local : le programme peut quand même ouvrir des .npy
            # sur un ordinateur qui n'a pas Picamera2.
            from picamera2 import Picamera2

            if self.fps <= 0:
                raise ValueError("La fréquence doit être positive.")
            if self.requested_width % 2 or self.requested_height % 2:
                raise ValueError("Avec YUV420, largeur et hauteur doivent être paires.")

            frame_period_us = int(round(1_000_000 / self.fps))
            if self.exposure_us > frame_period_us:
                raise ValueError(
                    "Le temps d'exposition dépasse la période entre deux images. "
                    "Diminue l'exposition ou le FPS."
                )

            self.status.emit("Initialisation de la caméra...")
            camera = Picamera2()

            available_controls = camera.camera_controls
            controls = {}

            if "AeEnable" in available_controls:
                controls["AeEnable"] = False

            if "ExposureTime" not in available_controls:
                raise RuntimeError("La caméra ne semble pas exposer le contrôle ExposureTime.")

            exposure_range = available_controls["ExposureTime"]
            if isinstance(exposure_range, (tuple, list)) and len(exposure_range) >= 2:
                min_exp, max_exp = int(exposure_range[0]), int(exposure_range[1])
                if not min_exp <= self.exposure_us <= max_exp:
                    raise ValueError(
                        f"Exposition demandée hors limites ({min_exp} à {max_exp} µs)."
                    )

            controls["ExposureTime"] = self.exposure_us
            if "FrameDurationLimits" in available_controls:
                controls["FrameDurationLimits"] = (frame_period_us, frame_period_us)
            else:
                self.status.emit(
                    "Attention : FrameDurationLimits indisponible; le FPS cible "
                    "ne peut pas être imposé par ce programme."
                )

            # YUV420 est moins volumineux que RGB888. On conserve seulement
            # le plan Y (luminance), qui est une image 2D de 8 bits.
            config = camera.create_video_configuration(
                main={
                    "size": (self.requested_width, self.requested_height),
                    "format": "YUV420",
                },
                buffer_count=3,
                queue=False,
                display=None,
                encode=None,
                controls=controls,
            )
            camera.configure(config)

            stream_config = camera.stream_configuration("main")
            width, height = map(int, stream_config["size"])

            # Estimation de l'espace nécessaire (images + timestamps).
            estimated_bytes = self.n_frames * height * width + self.n_frames * 8
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(self.filename.parent).free
            if estimated_bytes > free_bytes * 0.95:
                raise OSError(
                    "Espace disque insuffisant. Il faut environ "
                    f"{estimated_bytes / 1e9:.2f} Go, espace libre : "
                    f"{free_bytes / 1e9:.2f} Go."
                )

            # Création du .npy directement sur disque. Aucune liste d'images
            # n'est construite en RAM.
            images_map = np.lib.format.open_memmap(
                self.filename,
                mode="w+",
                dtype=np.uint8,
                shape=(self.n_frames, height, width),
            )
            times_map = np.lib.format.open_memmap(
                timestamps_path,
                mode="w+",
                dtype=np.float64,
                shape=(self.n_frames,),
            )

            camera.start()
            time.sleep(0.5)  # laisse le pipeline caméra se stabiliser
            self.status.emit(
                f"Acquisition en cours — {width} × {height}, YUV420 / plan Y 8 bits"
            )

            use_sensor_timestamp = None
    
            for i in range(self.n_frames):
                if self._stop_event.is_set():
                    break

                # L'image et ses métadonnées proviennent de la même requête.
                request = camera.capture_request()
                try:
                    yuv = request.make_array("main")
                    capture_metadata = request.get_metadata()
                finally:
                    # Important : toujours rendre le buffer à Picamera2.
                    request.release()

                # YUV420 : les premières 'height' lignes contiennent la
                # luminance Y. La tranche peut être une vue; l'affectation
                # vers le memmap fait la copie directement dans le fichier.
                gray = yuv[:height, :width]
                if gray.dtype != np.uint8:
                    raise TypeError(
                        f"Le flux YUV420 devait être uint8, reçu : {gray.dtype}."
                    )

                sensor_timestamp = capture_metadata.get("SensorTimestamp")
                if use_sensor_timestamp is None:
                    use_sensor_timestamp = sensor_timestamp is not None
                    if not use_sensor_timestamp:
                        timestamp_source = "time.perf_counter_ns (repli)"

                if use_sensor_timestamp and sensor_timestamp is not None:
                    timestamp_ns = int(sensor_timestamp)
                else:
                    timestamp_ns = time.perf_counter_ns()

                if first_timestamp_ns is None:
                    first_timestamp_ns = timestamp_ns

                images_map[i] = gray
                times_map[i] = (timestamp_ns - first_timestamp_ns) * 1e-9
                frames_written += 1

                if not first_capture_metadata:
                    first_capture_metadata = {
                        key: value.item() if isinstance(value, np.generic) else value
                        for key, value in capture_metadata.items()
                        if isinstance(value, (str, int, float, bool, np.generic))
                    }

                # N'émet pas un signal Qt à chaque image.
                if frames_written % 5 == 0 or frames_written == self.n_frames:
                    self.progress.emit(frames_written, self.n_frames)

            self.status.emit("Synchronisation des données vers le disque...")

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

        finally:
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass
                try:
                    camera.close()
                except Exception:
                    pass

            # Un flush final seulement : le faire à chaque image ralentirait
            # inutilement l'acquisition.
            flush_start = time.perf_counter()
            try:
                if images_map is not None:
                    images_map.flush()
                if times_map is not None:
                    times_map.flush()
            except Exception as exc:
                if not error_message:
                    error_message = f"Erreur pendant l'écriture finale : {exc}"
            flush_seconds = time.perf_counter() - flush_start

            # Fermer les mappings avant de réouvrir le fichier dans le visualiseur.
            del images_map
            del times_map

        elapsed = time.perf_counter() - start_wall
        fps_measured = None
        if frames_written > 1 and timestamps_path.exists():
            try:
                saved_times = np.load(timestamps_path, mmap_mode="r", allow_pickle=False)
                dt = float(saved_times[frames_written - 1] - saved_times[0])
                if dt > 0:
                    fps_measured = (frames_written - 1) / dt
                del saved_times
            except Exception:
                pass

        metadata = {
            "file": self.filename.name,
            "timestamps_file": timestamps_path.name,
            "status": "error" if error_message else (
                "stopped" if frames_written < self.n_frames else "complete"
            ),
            "frames_requested": self.n_frames,
            "frames_written": frames_written,
            "shape": [self.n_frames, height, width] if width and height else None,
            "stored_dtype": "uint8",
            "stored_pixels": "Y (luminance) plane extracted from YUV420; grayscale",
            "requested_fps": self.fps,
            "measured_average_fps": fps_measured,
            "exposure_requested_us": self.exposure_us,
            "main_stream": {
                "format": "YUV420",
                "size": [width, height] if width and height else None,
            },
            "timestamp_source": timestamp_source,
            "first_frame_metadata": first_capture_metadata,
            "elapsed_wall_seconds": elapsed,
            "final_flush_seconds": flush_seconds,
            "error": error_message or None,
            "note": (
                "Lire uniquement les frames [0:frames_written]. Si l'acquisition "
                "a été arrêtée avant la fin, les cases restantes du .npy ne sont pas valides."
            ),
        }

        try:
            if width and height:
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            if not error_message:
                error_message = f"Impossible d'écrire les métadonnées JSON : {exc}"

        if frames_written > 0:
            msg = (
                f"{frames_written}/{self.n_frames} images enregistrées. "
                f"Débit moyen mesuré : {fps_measured:.2f} fps. "
                f"Synchronisation finale : {flush_seconds:.2f} s."
                if fps_measured is not None
                else f"{frames_written}/{self.n_frames} images enregistrées."
            )
            if error_message:
                msg += f" Une erreur est survenue : {error_message}"
            self.completed.emit(str(self.filename), frames_written, msg)
        else:
            self.failed.emit(error_message or "Aucune image n'a été enregistrée.")


# ---------------------------------------------------------------------------
# Interface : acquisition d'abord, visualisation après
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acquisition caméra — NPY Memory Mapping")
        self.resize(1000, 760)

        self.capture_thread = None
        self.data = None  # np.memmap; une seule image est consultée à la fois
        self.metadata = {}
        self.valid_frames = 0
        self.current_filename = None

        central = QWidget()
        root = QVBoxLayout(central)

        # Paramètres d'acquisition
        acquisition_box = QGroupBox("Acquisition")
        form = QFormLayout(acquisition_box)

        self.n_frames = QSpinBox()
        self.n_frames.setRange(1, 1_000_000)
        self.n_frames.setValue(100)

        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.1, 240.0)
        self.fps.setDecimals(2)
        self.fps.setValue(10.0)

        self.exposure = QSpinBox()
        self.exposure.setRange(1, 1_000_000)
        self.exposure.setValue(1000)
        self.exposure.setSuffix(" µs")

        self.width = QSpinBox()
        self.width.setRange(64, 10000)
        self.width.setSingleStep(2)
        self.width.setValue(1920)

        self.height = QSpinBox()
        self.height.setRange(64, 10000)
        self.height.setSingleStep(2)
        self.height.setValue(1080)

        self.output_path = QLabel(str(Path.home() / "sequence.npy"))
        self.output_path.setWordWrap(True)
        self.choose_output_button = QPushButton("Choisir…")
        self.choose_output_button.clicked.connect(self.choose_output_file)
        path_row = QHBoxLayout()
        path_row.addWidget(self.output_path, 1)
        path_row.addWidget(self.choose_output_button)
        path_widget = QWidget()
        path_widget.setLayout(path_row)

        form.addRow("Nombre d'images :", self.n_frames)
        form.addRow("Fréquence cible :", self.fps)
        form.addRow("Exposition :", self.exposure)
        form.addRow("Largeur :", self.width)
        form.addRow("Hauteur :", self.height)
        form.addRow("Fichier .npy :", path_widget)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Démarrer l'acquisition")
        self.stop_button = QPushButton("Arrêter")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)

        self.progress_label = QLabel("Prêt. Aucune image n'est affichée pendant l'acquisition.")
        self.progress_label.setWordWrap(True)

        root.addWidget(acquisition_box)
        root.addLayout(buttons)
        root.addWidget(self.progress_label)

        # Visualiseur de fichiers .npy
        viewer_box = QGroupBox("Visualisation (après l'acquisition)")
        viewer_layout = QVBoxLayout(viewer_box)

        viewer_buttons = QHBoxLayout()
        self.open_button = QPushButton("Ouvrir un fichier .npy")
        self.open_button.clicked.connect(self.open_npy_file)
        self.frame_label = QLabel("Aucun fichier ouvert")
        viewer_buttons.addWidget(self.open_button)
        viewer_buttons.addWidget(self.frame_label, 1)

        self.image_label = QLabel("Ouvre un fichier .npy pour afficher une image.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 400)
        self.image_label.setStyleSheet("QLabel { background-color: #202020; color: white; }")

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.show_frame)

        nav = QHBoxLayout()
        self.previous_button = QPushButton("◀ Précédente")
        self.next_button = QPushButton("Suivante ▶")
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.previous_button.clicked.connect(self.previous_frame)
        self.next_button.clicked.connect(self.next_frame)
        nav.addWidget(self.previous_button)
        nav.addWidget(self.next_button)

        viewer_layout.addLayout(viewer_buttons)
        viewer_layout.addWidget(self.image_label, 1)
        viewer_layout.addWidget(self.frame_slider)
        viewer_layout.addLayout(nav)
        root.addWidget(viewer_box, 1)

        self.setCentralWidget(central)

    def choose_output_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer la séquence", str(Path.home() / "sequence.npy"),
            "Fichier NumPy (*.npy)"
        )
        if path:
            if not path.lower().endswith(".npy"):
                path += ".npy"
            self.output_path.setText(path)

    def start_acquisition(self):
        path = Path(self.output_path.text())
        if path.exists() or path.with_suffix(".json").exists() or path.with_name(
            path.stem + "_timestamps.npy"
        ).exists():
            answer = QMessageBox.question(
                self, "Écraser les fichiers?",
                "Un ou plusieurs fichiers de cette séquence existent déjà. Les remplacer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.capture_thread = AcquisitionThread(
            str(path), self.n_frames.value(), self.fps.value(),
            self.exposure.value(), self.width.value(), self.height.value()
        )
        self.capture_thread.progress.connect(self.update_progress)
        self.capture_thread.status.connect(self.progress_label.setText)
        self.capture_thread.completed.connect(self.acquisition_completed)
        self.capture_thread.failed.connect(self.acquisition_failed)
        self.capture_thread.finished.connect(self.thread_finished)

        self.set_acquisition_controls_enabled(False)
        self.stop_button.setEnabled(True)
        self.progress_label.setText("Démarrage de l'acquisition...")
        self.capture_thread.start()

    def stop_acquisition(self):
        if self.capture_thread is not None and self.capture_thread.isRunning():
            self.capture_thread.request_stop()
            self.stop_button.setEnabled(False)
            self.progress_label.setText("Arrêt demandé; fin de l'image en cours...")

    def update_progress(self, current: int, total: int):
        self.progress_label.setText(f"Acquisition : {current} / {total} images")

    def acquisition_completed(self, filename: str, count: int, message: str):
        self.progress_label.setText(message)
        self.open_npy_path(filename)
        if self.metadata.get("status") == "error":
            QMessageBox.warning(self, "Acquisition partielle", message)

    def acquisition_failed(self, message: str):
        self.progress_label.setText("Échec de l'acquisition.")
        QMessageBox.critical(self, "Erreur d'acquisition", message)

    def thread_finished(self):
        self.set_acquisition_controls_enabled(True)
        self.stop_button.setEnabled(False)

    def set_acquisition_controls_enabled(self, enabled: bool):
        for widget in (
            self.n_frames, self.fps, self.exposure, self.width, self.height,
            self.start_button, self.choose_output_button, self.open_button,
        ):
            widget.setEnabled(enabled)

    def open_npy_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une séquence", str(Path.home()),
            "Fichier NumPy (*.npy)"
        )
        if path:
            self.open_npy_path(path)

    def open_npy_path(self, filename: str):
        try:
            path = Path(filename)
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3:
                raise ValueError(
                    "Le fichier doit avoir la forme (images, hauteur, largeur). "
                    f"Forme reçue : {array.shape}."
                )
            if array.dtype != np.uint8:
                raise ValueError(
                    "Ce visualiseur attend des images uint8 en niveaux de gris. "
                    f"Type reçu : {array.dtype}."
                )

            metadata_path = path.with_suffix(".json")
            metadata = {}
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)

            valid_frames = int(metadata.get("frames_written", array.shape[0]))
            valid_frames = min(valid_frames, int(array.shape[0]))
            if valid_frames < 1:
                raise ValueError("Le fichier ne contient aucune image enregistrée.")

            self.data = array
            self.metadata = metadata
            self.valid_frames = valid_frames
            self.current_filename = str(path)

            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, valid_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.frame_slider.setEnabled(valid_frames > 1)
            self.previous_button.setEnabled(valid_frames > 1)
            self.next_button.setEnabled(valid_frames > 1)

            self.show_frame(0)

        except Exception as exc:
            QMessageBox.critical(self, "Impossible d'ouvrir le fichier", str(exc))

    def show_frame(self, index: int):
        if self.data is None or not self.valid_frames:
            return

        index = max(0, min(int(index), self.valid_frames - 1))
        frame = self.data[index]
        height, width = frame.shape

        # QImage doit posséder ses données : copy() évite de dépendre de la
        # durée de vie de la vue temporaire du fichier memory-mapped.
        qimage = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self.frame_label.setText(
            f"Image {index + 1}/{self.valid_frames} — {width} × {height} px"
        )
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)

    def previous_frame(self):
        self.frame_slider.setValue(max(0, self.frame_slider.value() - 1))

    def next_frame(self):
        self.frame_slider.setValue(
            min(self.valid_frames - 1, self.frame_slider.value() + 1)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.data is not None:
            self.show_frame(self.frame_slider.value())

    def closeEvent(self, event):
        if self.capture_thread is not None and self.capture_thread.isRunning():
            self.capture_thread.request_stop()
            self.capture_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
