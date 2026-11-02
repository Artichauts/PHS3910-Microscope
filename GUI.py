import sys
import os
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QGroupBox
)

# Import Picamera2 (spécifique Raspberry Pi OS)
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # Pour test hors Raspberry Pi si besoin


class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    recording_finished = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.running = True
        self.is_recording = False
        self.cam = None

    def run(self):
        if Picamera2 is None:
            self.status_message.emit("Erreur : Picamera2 n'est pas disponible sur ce système.")
            return

        try:
            self.cam = Picamera2()
            # Configuration en RGB888 pour la capture initiale de la caméra
            config = self.cam.create_video_configuration(main={"size": (1024, 768), "format": "RGB888"})
            self.cam.configure(config)
            
            # Appliquer l'exposition initiale si définie
            if self.settings.get("exposure_time"):
                self.cam.set_controls({"ExposureTime": int(self.settings["exposure_time"])})

            self.cam.start()
            self.status_message.emit("Caméra initialisée. Live en niveau de gris...")
            time.sleep(1) # Warm-up

            while self.running:
                if self.is_recording:
                    self.perform_recording()
                    self.is_recording = False
                else:
                    frame = self.cam.capture_array()
                    if frame is not None:
                        # Conversion de la frame RVB en niveau de gris uint8 (formule de luminance standard)
                        gray_frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                        self.frame_ready.emit(gray_frame)
                    time.sleep(0.03) # ~30 fps max pour le preview

        except Exception as e:
            self.status_message.emit(f"Erreur caméra : {str(e)}")
        finally:
            if self.cam:
                try:
                    self.cam.stop()
                    self.cam.close()
                except:
                    pass
            self.status_message.emit("Caméra fermée.")

    def perform_recording(self):
        folder_path = self.settings["folder_path"]
        filename = self.settings["filename"]
        num_frames = self.settings["num_frames"]
        interval = self.settings["interval"]
        
        os.makedirs(folder_path, exist_ok=True)
        output_path = os.path.join(folder_path, filename)
        temp_memmap_path = os.path.join(folder_path, "temp_memmap.dat")
        
        memmap_array = None
        self.status_message.emit(f"Enregistrement de {num_frames} frames en niveau de gris...")

        try:
            for i in range(num_frames):
                if not self.running:
                    break
                
                # Capture de l'image
                frame = self.cam.capture_array()
                
                # Conversion en niveau de gris uint8 (dimensions 2D : Hauteur x Largeur)
                gray_frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

                if memmap_array is None:
                    height, width = gray_frame.shape
                    shape = (num_frames, height, width)
                    memmap_array = np.memmap(temp_memmap_path, dtype=np.uint8, mode='w+', shape=shape)
                    total_size_mb = np.prod(shape) / 1e6
                    self.status_message.emit(f"Buffer disque initialisé : {shape} (Grayscale uint8) -> ~{total_size_mb:.2f} MB")

                # Écriture directe dans le buffer disque (sécurité RAM 2Go Pi4)
                memmap_array[i] = gray_frame
                memmap_array.flush()
                self.status_message.emit(f"Enregistrement frame {i+1}/{num_frames}")

                if interval > 0 and i < num_frames - 1:
                    time.sleep(interval)

            # Sauvegarde finale au format .npz avec les images en niveau de gris uint8 et les métadonnées
            self.status_message.emit("Compression et sauvegarde .npz...")
            np.savez(
                output_path,
                images=memmap_array,
                fps=self.settings["fps"],
                magnification=self.settings["magnification"],
                exposure_time=self.settings["exposure_time"],
                timestamp=time.time()
            )

            # Nettoyage du fichier temporaire memmap
            del memmap_array
            if os.path.exists(temp_memmap_path):
                os.remove(temp_memmap_path)

            self.recording_finished.emit(output_path)

        except Exception as e:
            self.status_message.emit(f"Erreur durant l'enregistrement : {str(e)}")
            if os.path.exists(temp_memmap_path):
                os.remove(temp_memmap_path)

    def trigger_record(self):
        self.is_recording = True

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pi4 - Capture Microscope Grayscale (.npz)")
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Panneau de Contrôle (Gauche) ---
        control_group = QGroupBox("Paramètres & Contrôles")
        control_layout = QGridLayout()

        control_layout.addWidget(QLabel("Dossier de sortie :"), 0, 0)
        self.input_folder = QLineEdit("./captures")
        control_layout.addWidget(self.input_folder, 0, 1)

        control_layout.addWidget(QLabel("Nom du fichier (.npz) :"), 1, 0)
        self.input_filename = QLineEdit("output_frames.npz")
        control_layout.addWidget(self.input_filename, 1, 1)

        control_layout.addWidget(QLabel("Nombre de frames :"), 2, 0)
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(1, 5000)
        self.spin_frames.setValue(50)
        control_layout.addWidget(self.spin_frames, 2, 1)

        control_layout.addWidget(QLabel("FPS :"), 3, 0)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 120.0)
        self.spin_fps.setValue(30.0)
        control_layout.addWidget(self.spin_fps, 3, 1)

        control_layout.addWidget(QLabel("Magnification (ex: 10x) :"), 4, 0)
        self.spin_mag = QDoubleSpinBox()
        self.spin_mag.setRange(0.1, 1000.0)
        self.spin_mag.setValue(1.0)
        control_layout.addWidget(self.spin_mag, 4, 1)

        control_layout.addWidget(QLabel("Temps d'expo (µs) :"), 5, 0)
        self.spin_exp = QSpinBox()
        self.spin_exp.setRange(100, 1000000)
        self.spin_exp.setValue(10000)
        control_layout.addWidget(self.spin_exp, 5, 1)

        control_layout.addWidget(QLabel("Intervalle (s) :"), 6, 0)
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.0, 10.0)
        self.spin_interval.setValue(0.0)
        control_layout.addWidget(self.spin_interval, 6, 1)

        self.btn_record = QPushButton("Lancer l'enregistrement")
        self.btn_record.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 10px;")
        self.btn_record.clicked.connect(self.start_recording)
        control_layout.addWidget(self.btn_record, 7, 0, 1, 2)

        # Indicateur visuel d'enregistrement clair
        self.lbl_record_indicator = QLabel("● STANDBY")
        self.lbl_record_indicator.setAlignment(Qt.AlignCenter)
        self.lbl_record_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: gray; background-color: #ecf0f1; border-radius: 5px; padding: 8px;")
        control_layout.addWidget(self.lbl_record_indicator, 8, 0, 1, 2)

        self.lbl_status = QLabel("Prêt.")
        control_layout.addWidget(self.lbl_status, 9, 0, 1, 2)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group, 1)

        # --- Affichage du Live Feed (Droite) ---
        preview_group = QGroupBox("Live Feed (Niveau de gris)")
        preview_layout = QVBoxLayout()
        
        self.lbl_preview = QLabel("Chargement du flux caméra...")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: black; color: white;")
        self.lbl_preview.setMinimumSize(640, 480)
        preview_layout.addWidget(self.lbl_preview)
        
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group, 2)

        self.blink_state = False
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.update_blink_indicator)

        self.camera_worker = CameraWorker(self.get_settings())
        self.camera_worker.frame_ready.connect(self.display_frame)
        self.camera_worker.recording_finished.connect(self.on_recording_finished)
        self.camera_worker.status_message.connect(self.set_status)
        self.camera_worker.start()

    def get_settings(self):
        return {
            "folder_path": self.input_folder.text(),
            "filename": self.input_filename.text(),
            "num_frames": self.spin_frames.value(),
            "fps": self.spin_fps.value(),
            "magnification": self.spin_mag.value(),
            "exposure_time": self.spin_exp.value(),
            "interval": self.spin_interval.value()
        }

    def start_recording(self):
        self.camera_worker.settings = self.get_settings()
        self.camera_worker.trigger_record()
        
        self.btn_record.setEnabled(False)
        self.lbl_record_indicator.setText("🔴 ENREGISTREMENT EN COURS")
        self.lbl_record_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: white; background-color: #e74c3c; border-radius: 5px; padding: 8px;")
        self.blink_timer.start(500)

    def update_blink_indicator(self):
        self.blink_state = not self.blink_state
        if self.blink_state:
            self.lbl_record_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c; background-color: #fadbd8; border-radius: 5px; padding: 8px;")
        else:
            self.lbl_record_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: white; background-color: #e74c3c; border-radius: 5px; padding: 8px;")

    def on_recording_finished(self, path):
        self.blink_timer.stop()
        self.btn_record.setEnabled(True)
        self.lbl_record_indicator.setText("● STANDBY")
        self.lbl_record_indicator.setStyleSheet("font-size: 16px; font-weight: bold; color: gray; background-color: #ecf0f1; border-radius: 5px; padding: 8px;")
        QMessageBox.information(self, "Succès", f"Enregistrement terminé !\nSauvegardé dans :\n{path}")

    def display_frame(self, frame):
        h, w = frame.shape
        bytes_per_line = w
        # Utilisation de QImage.Format_Grayscale8 pour un affichage direct en nuances de gris 8 bits
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_preview.setPixmap(pixmap.scaled(self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_status(self, message):
        self.lbl_status.setText(message)

    def closeEvent(self, event):
        self.camera_worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())