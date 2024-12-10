"""
by: Jose Carlo Sia / chaoserver

Inference Engine for Chicken Vocalization Analysis

Dependencies:
- torch: Deep learning framework for model operations
- numpy: Numerical computations and array operations
- sklearn.metrics: For calculating model performance metrics
- pathlib: For platform-independent path handling
- logging: For structured logging output

This module provides the core inference functionality for analyzing chicken vocalizations
using a pre-trained light-VGG11 model to detect distress calls.
"""

import torch
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from app.models import vgg11_bn
from .process_audio import AudioPreprocessor

# Configure logging with timestamp and severity level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class InferenceEngine:
    """
    Main engine for processing audio files and making predictions.
    
    Attributes:
        model: PyTorch model instance (VGG11)
        device: Computation device (CPU/GPU)
        preprocessor: Audio preprocessing component
    
    Dependencies:
        - Trained model file: ketexh-vocalization.pth
        - AudioPreprocessor: For converting audio to spectrograms
    """
    
    def __init__(self):
        self.model = None  # Lazy loading of model
        self.device = torch.device("cpu")  # Using CPU for deployment
        self.preprocessor = AudioPreprocessor()
        
    def load_model(self, model_path):
        """
        Loads and initializes the VGG11 model with trained weights.
        
        Args:
            model_path: Path to the trained model weights
            
        Security:
            Uses weights_only=True to prevent arbitrary code execution
            
        Raises:
            Exception if model loading fails
        """
        if self.model is None:
            try:
                logging.info(f"Creating VGG11 model...")
                self.model = vgg11_bn(num_classes=1)  # Binary classification
                logging.info(f"Loading weights from {model_path}")
                state_dict = torch.load(
                    model_path, 
                    map_location=self.device,
                    weights_only=True  # Security feature
                )
                self.model.load_state_dict(state_dict)
                self.model.eval()  # Set to evaluation mode
                logging.info("Model loaded successfully on CPU")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                logging.error(f"Model path exists: {Path(model_path).exists()}")
                raise

    async def process_audio_file(self, file_path: Path):
        """
        Processes an audio file to detect chicken distress calls.
        
        Process Flow:
        1. Load model if not loaded
        2. Preprocess audio into spectrograms
        3. Make predictions on each segment
        4. Calculate statistics and metrics
        5. Determine alert level
        
        Alert Thresholds:
        - HIGH: >70% distress calls
        - MODERATE: >50% distress calls
        - LOW: ≤50% distress calls
        
        Returns:
            Dictionary containing:
            - Segment counts and percentages
            - Alert level and recommendation
            - Evaluation metrics
        """
        try:
            # Model loading
            if self.model is None:
                self.load_model("/app/app/models/ketexh-vocalization.pth")
            
            # Audio preprocessing
            spectrograms = self.preprocessor.process_file(file_path)
            if spectrograms is None:
                raise ValueError("Failed to process audio file")
            
            # Batch prediction
            predictions = []
            confidences = []
            with torch.no_grad():  # Disable gradient computation for inference
                for spec in spectrograms:
                    x = spec.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                    output = self.model(x)  # Forward pass
                    prob = output.item()
                    pred = 1 if prob > 0.5 else 0  # Binary classification threshold
                    confidence = prob if pred == 1 else 1 - prob
                    predictions.append(pred)
                    confidences.append(confidence)
            
            # Statistical analysis
            total_segments = len(predictions)
            distress_segments = sum(predictions)
            barn_segments = total_segments - distress_segments
            avg_confidence = sum(confidences) / len(confidences)
            distress_percentage = (distress_segments / total_segments * 100)
            
            # Logging analysis summary
            logging.info("\n" + "="*50)
            logging.info("ANALYSIS SUMMARY")
            logging.info("="*50)
            logging.info(f"Total segments processed: {total_segments}")
            logging.info(f"Segments classified as barn sounds: {barn_segments} ({barn_segments/total_segments*100:.1f}%)")
            logging.info(f"Segments classified as distress calls: {distress_segments} ({distress_percentage:.1f}%)")
            logging.info(f"Average confidence: {avg_confidence:.1%}")
            
            # Calculate performance metrics
            predictions_array = np.array(predictions)
            accuracy = accuracy_score(predictions_array, predictions_array)
            precision = precision_score(predictions_array, predictions_array)
            recall = recall_score(predictions_array, predictions_array)
            f1 = f1_score(predictions_array, predictions_array)
            
            # Log metrics
            logging.info("\nEvaluation Metrics:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1-score: {f1:.4f}")
            logging.info("="*50)
            
            # Determine alert level based on thresholds
            if distress_percentage > 70:  # Critical threshold
                alert_level = "HIGH"
                recommendation = "Immediate farm inspection recommended"
            elif distress_percentage > 50:  # Warning threshold
                alert_level = "MODERATE"
                recommendation = "Schedule inspection within 24 hours"
            else:
                alert_level = "LOW"
                recommendation = "Normal conditions, maintain regular monitoring"
            
            # Prepare metrics for response
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
            # Compile final results
            result = {
                'total_segments': total_segments,
                'barn_segments': barn_segments,
                'distress_segments': distress_segments,
                'distress_percentage': distress_percentage,
                'average_confidence': avg_confidence,
                'alert_level': alert_level,
                'recommendation': recommendation,
                'evaluation_metrics': metrics
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            raise

# Singleton instance for application-wide use
engine = InferenceEngine()

async def process_audio_file(file_path: Path):
    """
    Wrapper function for processing audio files.
    Maintains single engine instance for efficiency.
    """
    return await engine.process_audio_file(file_path) 