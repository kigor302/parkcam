from picamera import PiCamera
from time import sleep

class Camera:
    def __init__(self):
        self.camera = PiCamera()

    def capture_image(self, image_path):
        self.camera.start_preview()
        sleep(2)  # Allow the camera to warm up
        self.camera.capture(image_path)
        self.camera.stop_preview()

    def stream_video(self):
        self.camera.start_preview()
        # Add code to stream video if needed
        sleep(10)  # Example duration for streaming
        self.camera.stop_preview()

    def close(self):
        self.camera.close()