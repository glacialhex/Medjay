# Medjay - Egyptian Hieroglyphs Recognition

AI-powered hieroglyph detection and classification using YOLO + CNN, running entirely in the browser.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔍 **Multi-Glyph Detection** - YOLO-based detection of multiple hieroglyphs in images
- 🧠 **AI Classification** - CNN classifier for 170 Gardiner sign classes
- 📸 **Live Camera Mode** - Real-time hieroglyph recognition
- 🖼️ **Image Upload** - Analyze photos of stone walls or papyrus
- ✨ **Stone Enhancement** - CLAHE preprocessing for worn/etched glyphs
- 🎯 **Interactive Feedback** - Correction system with pseudo-reinforcement learning
- 🔒 **Privacy-First** - All inference runs locally in browser (ONNX Runtime Web)

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+

### Installation

```bash
# Clone the repository
git clone https://github.com/glacialhex/Medjay.git
cd Medjay

# Set up Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download datasets
python3 combine_datasets.py  # Follow instructions to download

# Set up frontend
cd app
npm install
npm run dev  # Opens at http://localhost:5173
```

## Project Structure

```
Medjay/
├── app/                    # Vite frontend
│   ├── public/
│   │   ├── hieroglyph_model.onnx   # CNN classifier (38MB)
│   │   └── yolov8_model.onnx       # YOLO detector (12MB)
│   └── src/
│       ├── main.js         # App logic
│       ├── classifier.js   # ONNX inference
│       ├── detector.js     # YOLO detection
│       └── imageProcessor.js  # CLAHE preprocessing
├── datasets/               # Training data (gitignored)
├── runs/                   # Training outputs (gitignored)
├── train_classifier.py     # CNN training script
├── combine_datasets.py     # Dataset preparation
└── stonify_augment.py      # Stone texture augmentation
```

## Training

### Retrain Classifier

```bash
source venv/bin/activate
python3 train_classifier.py  # Trains on datasets/Combined_Hieroglyphs
```

Model auto-exports to `app/public/hieroglyph_model.onnx`

### Retrain YOLO Detector

```bash
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=datasets/hieroglyph-yolo-dataset/data.yaml \
  epochs=20 imgsz=640
```

Export: `yolo export model=runs/detect/trainX/weights/best.pt format=onnx`

## Current Status

⚠️ **See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed project history and known issues**

- **Classifier**: 87.45% validation accuracy (170 classes)
- **YOLO**: mAP50 = 0.60 (stone-trained) / 0.47 (original)
- **Known Issues**: Lower accuracy on real stone photos, see PROJECT_STATUS.md

## Datasets

This project uses:
- **Egyptian_hieroglyphs** (HuggingFace) - 4,210 images, 171 classes
- **Glyphnet** (Downloaded separately) - 17,379 images

Download instructions in `combine_datasets.py`

## Tech Stack

**Frontend:**
- Vite 7.3
- ONNX Runtime Web 1.23.2
- Vanilla JavaScript

**Backend (Training):**
- PyTorch 2.8.0
- Ultralytics YOLOv8
- OpenCV

## Contributing

This project is in active development. Key areas needing improvement:
1. Dataset expansion (need 20k+ images)
2. Better model architecture (ResNet vs SimpleCNN)
3. Multi-scale YOLO detection
4. Per-class accuracy evaluation

See `PROJECT_STATUS.md` for detailed improvement roadmap.

## License

MIT License - see LICENSE file

## Acknowledgments

- Gardiner Sign List for hieroglyph classification
- Glyphnet dataset (GAIA-IFAC-CNR)
- HuggingFace Egyptian_hieroglyphs dataset
- Ultralytics YOLOv8

## Contact

- GitHub: [@glacialhex](https://github.com/glacialhex)
- Repository: [Medjay](https://github.com/glacialhex/Medjay)
