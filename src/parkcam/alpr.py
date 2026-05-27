from fast_alpr import ALPR

class ALPRService:
    def __init__(self):
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v2-global-model",
        )

    def recognize_plate(self, image_path):
        alpr_results = self.alpr.predict(image_path)
        return alpr_results