import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
import sys
import warnings
import logging
import os
from fpdf import FPDF
import base64
import tempfile
from .model.model import ARMClassifier, UltrasoundClassifier # Import from local model file
from .utils import ensure_model_file
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_FILENAME_ULTRASOUND = "best_ultrasound_classifier.pth"
MODEL_FILENAME_ARM = "best_arm_classifier.pth"


def _normalize_state_dict_keys(state_dict, prefix="model."):
    """Ensure checkpoint keys match wrapped model attribute names."""
    normalized = {}
    for key, value in state_dict.items():
        # Remove Distributed/DataParallel prefix if present
        if key.startswith("module."):
            key = key[len("module."):]
        if not key.startswith(prefix):
            key = prefix + key
        normalized[key] = value
    return normalized


def load_models():
    """Load both ultrasound and ARM classifier models"""
    try:
        # Load Ultrasound Classifier
        ultrasound_model = UltrasoundClassifier()
        # Adjust path for backend structure
        current_dir = Path(__file__).parent.resolve()
        model_dir_candidates = [
            current_dir / 'output',
            current_dir.parent / 'backend' / 'output',
            current_dir.parent / 'output'
        ]

        ultrasound_model_path = ensure_model_file(
            MODEL_FILENAME_ULTRASOUND,
            model_dir_candidates,
            env_keys=['MODEL_ULTRASOUND_URL']
        )
        
        if not ultrasound_model_path.exists():
            raise FileNotFoundError(f"Ultrasound model file not found at {ultrasound_model_path}")
        
        ultrasound_state = torch.load(ultrasound_model_path, map_location=torch.device('cpu'))
        ultrasound_state = _normalize_state_dict_keys(ultrasound_state)
        ultrasound_model.load_state_dict(ultrasound_state, strict=False)
        ultrasound_model.eval()
        
        # Load ARM Classifier
        arm_model = ARMClassifier()
        arm_model_path = ensure_model_file(
            MODEL_FILENAME_ARM,
            model_dir_candidates,
            env_keys=['MODEL_ARM_URL']
        )
        
        arm_state = torch.load(arm_model_path, map_location=torch.device('cpu'))
        arm_state = _normalize_state_dict_keys(arm_state)
        arm_model.load_state_dict(arm_state, strict=False)
        arm_model.eval()
        
        return ultrasound_model, arm_model
        
    except Exception as e:
        # In a real backend, you might log this error instead of printing to st
        print(f"Failed to load models: {str(e)}") # Use print for backend logging
        return None, None

def predict_ultrasound(image, model):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
        class_names = ['non_ultrasound', 'ultrasound']
        predicted_class = class_names[pred.item()]
        
        non_ultrasound_prob = probs[0][0].item()
        ultrasound_prob = probs[0][1].item()
        
        # Add confidence threshold
        if conf.item() < 0.85:  # 85% confidence threshold
            predicted_class = "Uncertain"
        
        return predicted_class, conf.item(), non_ultrasound_prob, ultrasound_prob

def predict_arm(image, model):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
        class_names = ['ARM', 'Normal']
        predicted_class = class_names[pred.item()]
        
        arm_prob = probs[0][0].item()
        normal_prob = probs[0][1].item()
        
        # Add confidence threshold
        if conf.item() < 0.85:  # 85% confidence threshold
            predicted_class = "Uncertain"
        
        return predicted_class, conf.item(), arm_prob, normal_prob

# Preprocessing must match training
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = image.convert('RGB')
    img_tensor = preprocess_transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, image

def get_gradcam_visualization(model, image_tensor, original_image):
    try:
        # Assuming ModelWrapper is used or the model has a similar structure
        # Adjust target_layer access based on the actual model structure if needed
        if hasattr(model.model, 'layer4'):
             target_layer = model.model.layer4[-1].conv2
        else:
             # Fallback or raise error if the expected layer is not found
             print("Warning: Could not find expected target layer for Grad-CAM. Using the last convolutional layer instead.")
             # This requires knowing the last conv layer name for other model types
             # For simplicity, let's assume a resnet-like structure or require ModelWrapper
             # If using a different model without a .model attribute, you might need a different approach.
             # Let's stick to the assumption that the passed model is wrapped or ResNet-like for now.
             # If the structure is significantly different, this needs adjustment.
             # For a generic solution, you might need to inspect the model layers dynamically.
             # For now, let's assume the model either fits ModelWrapper or is a ResNet variant where layer4[-1].conv2 is valid.
             # If the model does not have a '.model' attribute (i.e., not wrapped by ModelWrapper),
             # try to access the layers directly assuming it's a torch.nn.Module.
             if isinstance(model, torch.nn.Module) and not hasattr(model, 'model'):
                 # Attempt to find a common last convolutional layer name or pattern
                 # This is a heuristic and might not work for all models.
                 # A more robust solution would involve inspecting model layers.
                 target_layer = None # Placeholder - needs actual logic to find the last conv layer
                 # Example heuristic for a simple CNN:
                 # for name, layer in model.named_modules():
                 #     if isinstance(layer, nn.Conv2d):
                 #         target_layer = layer

                 if target_layer is None:
                      print("Error: Could not automatically find a suitable layer for Grad-CAM visualization.")
                      return None # Cannot proceed without a target layer
             elif not hasattr(model, 'model'):
                  print("Error: Model structure not recognized for Grad-CAM visualization.")
                  return None

        # Assuming SimpleWrapper is used or not needed depending on how model is passed
        # If model passed to this function is already the unwrapped model, SimpleWrapper might be redundant.
        # Let's adjust based on the expectation that load_models returns raw models,
        # and ModelWrapper was used internally in app.py for GradCAM.
        # We should ideally wrap the model here if needed for GradCAM.
        
        # Re-implement SimpleWrapper if needed or adjust GradCAM call
        # Based on original app.py, SimpleWrapper was defined and used inside this function.
        # Let's keep it consistent.
        class SimpleWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)
        
        wrapped_model = SimpleWrapper(model)
        wrapped_model.eval()

        if target_layer is None: # Check again if heuristic failed
             print("Error: Target layer for Grad-CAM is None.")
             return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Pass the wrapped_model to GradCAM
            cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
        
        grayscale_cam = cam(input_tensor=image_tensor)
        grayscale_cam = grayscale_cam[0, :]
        
        original_np = np.array(original_image.convert('RGB'))
        original_np = cv2.resize(original_np, (224, 224))
        original_np = original_np.astype(np.float32) / 255.0
        
        visualization_np = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
        visualization_pil = Image.fromarray(visualization_np)
        resized_visualization_pil = visualization_pil.resize((224, 224))
        return resized_visualization_pil
        
    except Exception as e:
        print(f"Visualization could not be generated: {str(e)}") # Use print for backend logging
        return None

def create_pdf_report(original_image, gradcam_image, prediction_results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.cell(200, 10, txt="ARM Detection Report", ln=1, align='C')
    pdf.ln(10)
    
    # Add Original Image
    pdf.cell(0, 10, txt="Original Image:", ln=1)
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_file.close()  # Close the file handle
    try:
        original_image.save(temp_file.name, format="PNG")
        pdf.image(temp_file.name, x=None, y=None, w=100)
    finally:
        try:
            os.unlink(temp_file.name)  # Delete the temporary file
        except:
            pass  # Ignore any errors during cleanup
    pdf.ln(10)
    
    # Add Grad-CAM Visualization
    if gradcam_image:
        pdf.cell(0, 10, txt="Grad-CAM Heatmap:", ln=1)
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.close()  # Close the file handle
        try:
            gradcam_image.save(temp_file.name, format="PNG")
            pdf.image(temp_file.name, x=None, y=None, w=100)
        finally:
            try:
                os.unlink(temp_file.name)  # Delete the temporary file
            except:
                pass  # Ignore any errors during cleanup
        pdf.ln(10)
    
    # Add Prediction Results
    pdf.cell(0, 10, txt="Prediction Results:", ln=1)
    pdf.ln(5)
    
    # Ultrasound Classification Results
    pdf.cell(0, 10, txt=f"Image Type: {prediction_results['image_type']}", ln=1)
    pdf.cell(0, 10, txt=f"Non-Ultrasound Probability: {prediction_results['non_ultrasound_prob']:.2%}", ln=1)
    pdf.cell(0, 10, txt=f"Ultrasound Probability: {prediction_results['ultrasound_prob']:.2%}", ln=1)
    pdf.ln(5)
    
    # ARM Classification Results (if applicable)
    if prediction_results['is_ultrasound']:
        pdf.cell(0, 10, txt="ARM Classification:", ln=1)
        pdf.cell(0, 10, txt=f"  Classification: {prediction_results['arm_class']}", ln=1)
        if prediction_results['arm_confidence'] is not None:
             pdf.cell(0, 10, txt=f"  Confidence: {prediction_results['arm_confidence']:.2%}", ln=1)
        if prediction_results['arm_prob'] is not None:
             pdf.cell(0, 10, txt=f"  ARM Probability: {prediction_results['arm_prob']:.2%}", ln=1)
        if prediction_results['normal_prob'] is not None:
             pdf.cell(0, 10, txt=f"  Normal Probability: {prediction_results['normal_prob']:.2%}", ln=1)
    else:
        pdf.cell(0, 10, txt="ARM Classification: Not Applicable (Image not an ultrasound)", ln=1)
        
    pdf.ln(10)
    
    # Important Notes
    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(0, 10, txt="Important Notes:", ln=1)
    pdf.set_font("Arial", size=10)
    # Use a fixed width slightly less than available space and add a small right margin
    available_width = pdf.w - pdf.l_margin - pdf.r_margin
    multi_cell_width = available_width - 5 # Subtract 5mm for right margin
    
    pdf.multi_cell(multi_cell_width, 5, txt="This is an AI-assisted diagnosis and should be verified by a medical professional.", align='J', border=0)
    pdf.multi_cell(multi_cell_width, 5, txt="The results are based on visual analysis of the ultrasound image. Clinical correlation and additional diagnostic tests may be required.", align='J', border=0)

    # Output PDF as bytes using the correct destination parameter
    return pdf.output(dest='S').encode('latin-1') 