# Hieroglyphics Identifier Project - Complete Status Report

## Project Overview
Browser-based Egyptian hieroglyph recognition system using YOLO (detection) + CNN (classification). Built with Vite, ONNX Runtime Web, and PyTorch.

**Repository**: https://github.com/glacialhex/Medjay  
**Local Dev**: http://localhost:5173/

---

## Initial State (Early Success)

### What Worked Initially
- **Classifier accuracy**: ~99% on validation set
- **Dataset**: Glyphnet (17,379 images, clean/drawn glyphs)
- **Model**: Simple CNN (3 conv layers + 2 FC layers)
- **Testing**: Single, clean hieroglyph images worked perfectly

### Why It Worked
- Clean, consistent training data (all drawn/printed glyphs)
- Single glyph per image (no detection needed)
- Good class balance in Glyphnet dataset
- Simple preprocessing (resize + normalize)

---

## What Went Wrong

### Issue #1: Stone Wall Detection Failure
**Problem**: Model failed completely on real stone wall photos (etched hieroglyphs)
- YOLO couldn't detect individual glyphs on textured backgrounds
- Classifier saw entire wall as one image → misclassified as "N35 Ripple of Water"

**Root Cause**: Training data mismatch
- Trained on: Clean, white-background drawings
- Tested on: Low-contrast stone carvings with texture

### Issue #2: Accuracy Degradation After Stone Training
**Problem**: After adding stone-texture augmentation, accuracy dropped from 99% → 50-70%
- "Stonified" training made model worse at clean images
- Overly aggressive augmentation (noise, blur, shadows)

**What Was Done**:
1. Created `stonify_augment.py` - adds stone textures to training images
2. Retrained YOLO on augmented data → mAP50: 0.60
3. Retrained classifier with heavy augmentation → 77% validation accuracy

**Result**: Model became better at noise but worse at everything else

### Issue #3: Live Camera Misclassification
**Problem**: After retraining cycles, live camera classified everything as "X8 Conical Loaf"
- Model became biased toward certain classes
- Possible label corruption during multiple retraining sessions

### Issue #4: Missing Basic Glyphs
**Problem**: User tested with ankh (S34) and hand (D46) - both failed
- Dataset didn't have enough coverage of common glyphs
- Glyphnet focused on specific temple inscriptions, not comprehensive Gardiner list

---

## What Has Been Implemented

### ✅ Core Features
1. **Two-Stage Pipeline**: YOLO detection → CNN classification
2. **Browser-Based Inference**: ONNX Runtime Web (privacy-preserving)
3. **Multi-Glyph Support**: Detects and classifies multiple glyphs in one image
4. **Camera Mode**: Live hieroglyph recognition via webcam (mirrored)

### ✅ Data & Training
1. **Combined Dataset**: 
   - Egyptian_hieroglyphs (4,210 images, 171 classes)
   - Glyphnet subset (used for YOLO)
   - **Total**: 17,716 images, 171 classes (FIXED - was 3,584)
2. **Latest Model Performance**:
   - Classifier: 87.45% validation accuracy
   - YOLO: mAP50 = 0.60 (stone-trained)
3. **Training Scripts**:
   - `train_classifier.py` - CNN training
   - `train_detector.py` - YOLOv8 training
   - `stonify_augment.py` - Stone texture augmentation
   - `combine_datasets.py` - Dataset merging

### ✅ UI Features
1. **Upload & Camera Modes**: Switch between image upload and live camera
2. **Enhance for Stone Button**: CLAHE + sharpening preprocessing
3. **Interactive Feedback System**:
   - ✓ Correct / ✗ Wrong buttons on each prediction
   - Correction dialog with glyph search
   - Pseudo-reinforcement learning (adjusts probabilities based on corrections)
4. **Results Display**: Top predictions with confidence, Gardiner codes, meanings

### ✅ Infrastructure
- Git repository with version control
- Proper `.gitignore` (excludes venv, datasets, runs)
- GitHub deployment
- LocalStorage for user corrections/feedback

---

## Current Problems

### 🔴 Critical Issues

1. **Accuracy Still Too Low** (87% validation, likely lower in practice)
   - User reports "huge error margin"
   - Basic glyphs like ankh/hand still misclassified
   - Gap between validation accuracy and real-world performance

2. **Detection on Stone Walls** (Still Unreliable)
   - YOLO struggles with low-contrast etched glyphs
   - Stone texture interferes with bounding box detection
   - Multi-scale detection not implemented

3. **~~Training Data Quality Issues~~** (FIXED)
   - ~~Combined dataset has only 3,584 images (should have ~20k+)~~
   - Now: 17,716 images combined properly
   - ~~Glyphnet has 17,379 images but wasn't properly integrated~~
   - Class imbalance likely (some glyphs have <10 examples)

### ⚠️ Medium Issues

4. **No Evaluation Metrics Tracked**
   - Don't know per-class accuracy
   - Don't know which glyphs fail most
   - No confusion matrix

5. **Model Architecture May Be Too Simple**
   - 3-layer CNN from 2015-era design
   - No residual connections, batch norm, attention
   - Modern alternatives: ResNet, EfficientNet, Vision Transformer

---

## What Needs To Be Done

### Priority 1: Fix Accuracy (IMMEDIATE)

#### Option A: Get More Data (Recommended)
```bash
# Download additional datasets from Kaggle
# Already have: Egyptian_hieroglyphs (4,210)
# Need: GlyphDataset (17k), Hieroglyphs_Dataset (4k)
# Target: 25k+ images, 171 classes
```

#### Option B: Fix Data Integration
- **Problem**: Glyphnet has 17k images but only 3.5k were combined
- **Action**: Debug `combine_datasets.py` - Glyphnet path is wrong
- **Expected**: Should have 21k+ images total

#### Option C: Better Model Architecture
```python
# Replace SimpleCNN with:
# - ResNet18 (pretrained on ImageNet, fine-tuned)
# - Or: EfficientNet-B0
# - Or: ConvNeXt-Tiny
```

### Priority 2: Detection Improvements

1. **Multi-Scale Detection** (from implementation_plan.md)
   - Run YOLO at 640px, 480px, 320px
   - Merge detections with NMS
   - Catches both large and small glyphs

2. **Edge-Based Preprocessing**
   - Add Sobel/Canny edge detection
   - Helps with low-contrast stone carvings

3. **Better Augmentation**
   - Current stonify is too aggressive
   - Need more subtle: light shadows, slight texture, not full noise

### Priority 3: Evaluation & Debugging

1. **Add Metrics Dashboard**
   ```python
   # Track in training:
   # - Per-class accuracy
   # - Confusion matrix
   # - Top-5 accuracy
   # - Failed examples (save to disk)
   ```

2. **Test Set Evaluation**
   - Create held-out test set (never seen during training)
   - Run inference, save results
   - Manual review of failures

### Priority 4: Features (Lower Priority)

1. **Dictionary/Translation** (Phase 2 from original plan)
   - Group nearby glyphs into words
   - Phonetic lookup
   - Middle Egyptian dictionary

2. **WebGPU Acceleration**
   - Switch from WASM to WebGPU backend
   - 5-10x speedup

3. **PWA/Offline Mode**
   - Service worker for offline use

---

## Technical Debt

1. **No Unit Tests**: Zero test coverage
2. **No CI/CD**: Manual git push only
3. **Hard-Coded Paths**: Training scripts have absolute paths
4. **No Logging**: Console.log only, no structured logging
5. **No Error Handling**: Many try-catch blocks just alert()

---

## Files & Structure

```
hieroglyphics-identifier/
├── app/                          # Vite frontend
│   ├── public/
│   │   ├── hieroglyph_model.onnx   # CNN (38MB, 170 classes)
│   │   └── yolov8_model.onnx       # YOLO (11.7MB)
│   ├── src/
│   │   ├── main.js                 # Main app logic
│   │   ├── classifier.js           # ONNX inference + reinforcement
│   │   ├── detector.js             # YOLO inference + NMS
│   │   ├── imageProcessor.js       # CLAHE preprocessing
│   │   ├── gardinerSigns.js        # Gardiner Sign List data
│   │   └── style.css
│   └── index.html
├── datasets/
│   ├── Egyptian_hieroglyphs/       # 4,210 images (downloaded)
│   ├── Glyphnet/                   # 17,379 images (not fully integrated)
│   ├── Combined_Hieroglyphs/       # 17,716 images (FIXED - was 3,584)
│   └── hieroglyph-yolo-dataset/    # YOLO bounding boxes
├── runs/                           # Training outputs
│   └── detect/
│       ├── train7/                 # Original YOLO (mAP50: 0.47)
│       └── train_stone2/           # Stone YOLO (mAP50: 0.60)
├── train_classifier.py             # ⚠️ Points to Combined_Hieroglyphs
├── train_detector.py               # YOLO training (not used recently)
├── stonify_augment.py              # Stone texture augmentation
├── combine_datasets.py             # ⚠️ BUG: Doesn't find Glyphnet properly
└── venv/                           # Python 3.9 environment
```

---

## Key Insights for Next Agent

### What Worked
- ONNX Runtime Web is fast and works great
- YOLO + CNN pipeline is correct approach
- User feedback system is well-designed
- CLAHE preprocessing helps a bit

### What Didn't Work
- Heavy augmentation hurt more than helped
- Multiple retraining passes corrupted model
- Stone-specific training reduced general accuracy
- Dataset size is too small (3.5k vs needed 20k+)

### Biggest Bottleneck
**FIXED**: Dataset now has 17,716 images (was 3,584)
- Previous: 3,584 images (only 21 images/class on average)
- Now: 17,716 images (103+ images/class average)
- Next step: Retrain with `python3 train_classifier.py`

### Quick Wins
1. **Fix combine_datasets.py** to use all of Glyphnet → instant 6x data increase
2. **Download GlyphDataset from Kaggle** → +17k images
3. **Use pre-trained ResNet** instead of SimpleCNN → better features

### Red Flags
- User said "huge error margin" even after latest retraining (87% val acc)
- This suggests validation set may not match real-world use
- Need to separate clean-glyph validation vs stone-photo validation
- Possible data leakage or overfitting

---

## Reproduction Steps (For Next Agent)

### Setup
```bash
cd /Users/yousef/.gemini/antigravity/scratch/hieroglyphics-identifier
source venv/bin/activate
cd app && npm run dev  # Dev server on :5173
```

### Retrain Classifier
```bash
# Current (buggy):
python3 train_classifier.py  # Uses Combined_Hieroglyphs (3.5k images)

# After fixing combine_datasets.py:
python3 combine_datasets.py  # Should get 21k images
python3 train_classifier.py  # Retrain on full dataset
```

### Retrain YOLO
```bash
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=datasets/hieroglyph-yolo-dataset/data.yaml \
  epochs=20 imgsz=640 workers=0
```

### Export Models
```bash
# Classifier (auto-exported by train_classifier.py)
# YOLO (manual):
yolo export model=runs/detect/trainX/weights/best.pt format=onnx
cp runs/detect/trainX/weights/best.onnx app/public/yolov8_model.onnx
```

---

## Recommendations

### Immediate (Do First)
1. ~~Fix `combine_datasets.py` to properly merge Glyphnet~~ ✅ DONE (17,716 images now)
2. Retrain with full 21k dataset: `python3 train_classifier.py`
3. Add validation split logging (per-class accuracy)

### Short-Term (Next Session)
4. Download GlyphDataset from Kaggle
5. Switch to ResNet18 architecture
6. Implement per-class accuracy tracking

### Long-Term (After Accuracy Fixed)
7. Multi-scale YOLO detection
8. Dictionary/translation feature
9. WebGPU acceleration

---

## Contact & Handoff Notes

- **GitHub**: https://github.com/glacialhex/Medjay
- **Local Path**: `/Users/yousef/.gemini/antigravity/scratch/hieroglyphics-identifier`
- **Python**: 3.9 in venv
- **Node**: Vite 7.3.1
- **Key Libraries**: PyTorch 2.8.0, ultralytics 8.3.251, onnxruntime-web 1.23.2

**Current Model Files**:
- `app/public/hieroglyph_model.onnx` - 38MB, 170 classes, 87% val acc
- `app/public/yolov8_model.onnx` - 11.7MB, train7 (not stone version)

**Critical Bug ~~Found~~ FIXED**: `combine_datasets.py` line 11 - `GLYPHNET` path has been corrected:
```python
GLYPHNET = Path("datasets/Glyphnet/Automated_extracted")  # Fixed - was /Automated_extracted/Automated
```

**User Feedback**: "huge error margin" - indicates 87% is not enough, aim for 95%+

Good luck! 🙏
