import unittest
from parkcam.alpr import ALPR

class TestALPR(unittest.TestCase):
    def setUp(self):
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v2-global-model",
        )

    def test_predict(self):
        results = self.alpr.predict("test_image.png")
        self.assertIsNotNone(results)
        self.assertIsInstance(results, dict)  # Assuming results are returned as a dictionary

if __name__ == '__main__':
    unittest.main()