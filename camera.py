
from picamera2 import Picamera2
import cv2
from time import sleep


def capture_one_frame()
    cam = Picamera2()

    cam.configure(
        cam.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
    )

    cam.start()
    sleep(2)

    frame = cam.capture_array()

    if frame is not None:
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if cv2.imwrite("/home/phs3910/test.jpg", frame):
            print("Image saved successfully!")
        else:
            print("Could not save image")
    else:
        print("Could not capture image")

    cam.stop()
    return frame
