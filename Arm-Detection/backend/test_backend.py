# backend/test_backend.py
import unittest
import os
import torch
from model.model import UltrasoundClassifier, ARMClassifier

class TestBackend(unittest.TestCase):
    def setUp(self):
        self.stage1_model_path = "output/best_ultrasound_classifier.pth"
        self.stage2_model_path = "output/best_arm_classifier.pth"

    def test_model_loading(self):
        # Test Stage 1 model loading
        if not os.path.exists(self.stage1_model_path):
            self.skipTest("Stage 1 model file not found")
        model1 = UltrasoundClassifier()
        model1.load_state_dict(torch.load(self.stage1_model_path, map_location="cpu"))
        self.assertIsNotNone(model1)

        # Test Stage 2 model loading
        if not os.path.exists(self.stage2_model_path):
            self.skipTest("Stage 2 model file not found")
        model2 = ARMClassifier()
        model2.load_state_dict(torch.load(self.stage2_model_path, map_location="cpu"))
        self.assertIsNotNone(model2)

if __name__ == "__main__":
    unittest.main()