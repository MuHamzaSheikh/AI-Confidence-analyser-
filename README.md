# AI Confidence Analyzer

An AI-powered confidence estimation system that combines **facial emotion analysis** and **voice characteristics** to generate an overall confidence score from a short real-time session.

> **Note:** This project provides an AI-based *estimation* from observable visual and audio cues. It is not a clinical or scientifically validated measurement of a person's psychological confidence.

## Overview

The AI Confidence Analyzer captures webcam video and microphone audio during a short session. It analyzes facial expressions and audio characteristics independently, converts those signals into scores, and combines them into a final confidence estimate.

The project was built to explore how **computer vision, facial emotion recognition, digital signal processing, and machine learning** can be combined into a practical AI application.

## How It Works

```text
Webcam ──► Facial Emotion Analysis ──► Visual Score ──┐
                                                       ├──► Final Confidence Estimate
Microphone ──► Audio Feature Analysis ──► Audio Score ─┘
```

### 1. Visual Analysis

The application reads webcam frames and uses **DeepFace** for emotion analysis. Detected emotion probabilities are mapped to weighted values and aggregated into a visual score.

### 2. Audio Analysis

The user's voice is recorded during the session and analyzed with **Librosa**. The current implementation uses audio energy and spectral flatness to estimate vocal stability.

### 3. Score Fusion

The visual and audio scores are combined into one final estimate. The current implementation weights the visual component at **60%** and the audio component at **40%**.

## Key Features

- 🎥 Real-time webcam capture
- 🙂 Facial emotion analysis with DeepFace
- 🎙️ Microphone recording and audio analysis
- 📊 Visual + audio confidence score fusion
- ⏱️ Timed analysis session
- 🧠 Machine-learning model included in the project
- 🐍 Python-based implementation

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Facial Analysis | DeepFace |
| Audio Processing | Librosa, PyAudio |
| Numerical Computing | NumPy |
| ML Model | TensorFlow/Keras (`.h5`) |
| Data | Dataset + Excel analysis report |

## Project Structure

```text
AI-Confidence-analyser-/
├── main.py                  # Real-time confidence analysis
├── main_custom.py           # Custom analysis workflow
├── gui_main.py              # GUI-based interface
├── hud_main.py              # HUD-style interface
├── create_dataset.py        # Dataset preparation
├── check_count.py           # Dataset/count utility
├── excels_column.py         # Dataset/report utility
├── my_confidence_model.h5   # Trained model
├── dataset/                 # Project dataset
├── requirement.txt          # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.x
- Webcam
- Microphone
- Required Python packages listed in `requirement.txt`

### Installation

```bash
git clone https://github.com/MuHamzaSheikh/AI-Confidence-analyser-.git
cd AI-Confidence-analyser-
pip install -r requirement.txt
```

### Run

The main real-time workflow can be started with:

```bash
python main.py
```

The application starts the webcam and microphone analysis and runs a timed confidence-checking session.

## Scoring Approach

The current implementation calculates:

- **Visual score:** derived from facial emotion probabilities using predefined emotion weights.
- **Audio score:** derived from voice energy and spectral characteristics.
- **Final score:** a weighted combination of the two components.

The implementation currently uses:

```text
Final Score = (Visual Score × 0.60) + (Audio Score × 0.40)
```

## Why This Project?

Confidence is expressed through multiple observable signals rather than a single measurement. This project explores a multimodal AI approach by combining visual and audio information instead of relying on only one data source.

It also provided hands-on experience with:

- Real-time computer vision
- Facial emotion recognition
- Audio feature extraction
- Model integration
- Dataset preparation
- Multithreaded audio recording
- Building an end-to-end AI application

## Future Improvements

Potential improvements include:

- Replace heuristic scoring with a properly validated multimodal model
- Add stronger speech/prosody features such as pitch, tempo, pauses, and MFCCs
- Improve face detection and multi-face handling
- Add model evaluation metrics and a dedicated test set
- Add confidence calibration and uncertainty estimates
- Build a modern web/mobile interface
- Add experiment tracking and reproducible model training
- Deploy the inference service using cloud infrastructure such as AWS

## Disclaimer

This project is intended for **educational and experimental purposes**. Its confidence score is an algorithmic estimate based on selected visual and audio features and should not be interpreted as a psychological, medical, hiring, or personality assessment.

## Author

**Muhammad Hamza Sheikh**

- GitHub: https://github.com/MuHamzaSheikh
- Portfolio: https://mhsportfolio-nu.vercel.app/
