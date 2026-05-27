from parkcam.camera import Camera
from parkcam.alpr import ALPR
from parkcam.utils import load_config

def main():
    config = load_config()
    camera = Camera(config['camera'])
    alpr = ALPR(
        detector_model=config['alpr']['detector_model'],
        ocr_model=config['alpr']['ocr_model'],
    )

    while True:
        image = camera.capture_image()
        results = alpr.predict(image)
        print(results)

if __name__ == "__main__":
    main()