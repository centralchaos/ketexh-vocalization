# Chicken Vocalization Analysis API

## Overview
This project provides an API service for analyzing chicken vocalizations to detect distress calls. It uses a deep learning model (light-VGG11) trained on spectrogram data to classify audio segments as either normal barn sounds or distress calls.

Developed by: Jose Carlo Sia / chaoserver
Model: KE Tech Vocalization Detector (ketexh-vocalization.pth)
Training data and training method adopted from Axio Mao on Automated Chicken Vocalization Detection

## Features
- Real-time audio analysis
- Distress call detection
- Confidence scoring
- Alert level classification
- Detailed metrics and recommendations

## System Requirements
- Docker and Docker Compose
- Python 3.9+
- Storage for audio processing
- Network connectivity for API access

## Quick Start

### 1. Clone the Repository 

bash
git clone https://github.com/yourusername/chicken-vocalization.git
cd chicken-vocalization


### 2. Model Setup
Download the pre-trained model:
- Model: ketexh-vocalization.pth
- Place in: `app/models/`
- Trained by: KE Texh / chaoserver
- Specifications:
  - Architecture: light-VGG11 with batch normalization
  - Input: Mel spectrograms (88x87)
  - Output: Binary classification (normal/distress)

### 3. Build and Run


Build Docker containers:

docker-compose up --build
API will be available at http://localhost:1200

## Directory Structure

project_root/
├── app/
│ ├── api/ # API endpoints and models
│ ├── core/ # Core processing logic
│ └── models/ # Neural network models
├── raspberry_pi/ # Client application
│ ├── audio_files/ # Input directory
│ ├── processed_audio/ # Processed files
│ └── results/ # Analysis results
└── docker/ # Docker configuration

## API Endpoints

### 1. Submit Audio for Analysis

bash
POST /api/v1/analyze
Content-Type: multipart/form-data

bash
POST /api/v1/analyze
Content-Type: multipart/form-data


### 2. Check Analysis Status

bash
GET /api/v1/status/{job_id}

### 3. Health Check

bash
GET /api/v1/health

## Client Usage
The Raspberry Pi client monitors an audio directory:

bash
cd raspberry_pi
python3 send_audio.py


Configuration:
- Input: `audio_files/`
- Processed: `processed_audio/`
- Results: `results/`

## Audio Processing Pipeline

1. **Preprocessing**
   - Sample Rate: 44.1kHz
   - Segment Duration: 1 second
   - FFT Window: 2048 points
   - Hop Length: 512 samples
   - Mel Bands: 88
   - Frequency Range: 2048-4096 Hz

2. **Model Inference**
   - Batch processing of segments
   - Confidence scoring
   - Threshold-based classification

3. **Alert Levels**
   - HIGH: >70% distress calls
   - MODERATE: >50% distress calls
   - LOW: ≤50% distress calls

## Performance Metrics
The system provides:
- Accuracy
- Precision
- Recall
- F1 Score
- Confidence scores
- Segment statistics

## Security Features
- Read-only application code
- Secure model loading (weights_only)
- Temporary file cleanup
- Input validation

## Logging
- API logs: Container logs
- Client logs: audio_sender.log
- Detailed processing metrics
- Error tracking

## Development Notes
- Built with FastAPI for async processing
- Docker for consistent deployment
- Modular design for easy updates
- Extensive error handling

## Acknowledgments
- KE Texh for the vocalization model
- VGG11 architecture authors
- Axio Mao for the training data and training method
- Axio Mao for the light-VGG11 architecture
- FastAPI framework

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Jose Carlo Sia (chaoserver)
KE Texh Research



