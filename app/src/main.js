import './style.css';
import { classifyImage, loadModel, getModelStatus, isModelDemo } from './classifier';
import { gardinerSigns, categories, searchSigns } from './gardinerSigns';
import { detectGlyphs, loadDetector } from './detector';
import { enhanceImage } from './imageProcessor';

// App state
let currentMode = 'upload'; // 'upload' or 'camera'
let isEnhanced = false;
let cameraStream = null;
let isProcessing = false;
let animationFrameId = null;

// Initialize the app
function init() {
  renderApp();
  loadModel(); // Pre-load the model
}

// Main render function
function renderApp() {
  document.querySelector('#app').innerHTML = `
    <header class="header">
      <div class="logo">
        <span class="logo-icon">𓂀</span>
        <h1 class="title">Hieroglyphics Identifier</h1>
      </div>
      <p class="subtitle">AI-powered Ancient Egyptian hieroglyph recognition using the Gardiner Sign List</p>
    </header>
    
    <main class="main-container">
      <section class="card input-card">
        <div class="card-header">
          <div class="card-icon">📷</div>
          <h2 class="card-title">Input Source</h2>
        </div>
        
        <div class="mode-toggle">
          <button class="mode-btn ${currentMode === 'upload' ? 'active' : ''}" id="uploadModeBtn">
            <span>📁</span> Upload Image
          </button>
          <button class="mode-btn ${currentMode === 'camera' ? 'active' : ''}" id="cameraModeBtn">
            <span>🎥</span> Live Camera
          </button>
        </div>
        
        <div id="inputArea">
          ${currentMode === 'upload' ? renderUploadArea() : renderCameraArea()}
        </div>
        
        <div id="previewArea"></div>
      </section>
      
      <section class="card results-container">
        <div class="card-header">
          <div class="card-icon">𓊹</div>
          <h2 class="card-title">Identification Results</h2>
        </div>
        
        <div id="resultsArea">
          ${renderEmptyResults()}
        </div>
      </section>
    </main>
    
    <section class="card" style="max-width: 1400px; margin: 40px auto;">
      <div class="card-header">
        <div class="card-icon">📊</div>
        <h2 class="card-title">Dataset & Setup Information</h2>
      </div>
      <div id="datasetInfo">
        ${renderDatasetInfo()}
      </div>
    </section>
    
    <footer class="footer">
      <p>Built with TensorFlow.js • Trained on Glyphnet dataset • <a href="https://github.com/GAIA-IFAC-CNR/Glyphnet" target="_blank">View Research</a></p>
      <p style="margin-top: 8px; font-size: 0.85rem;">Based on the Gardiner Sign List classification system</p>
    </footer>
  `;

  attachEventListeners();
}

// Render upload area
function renderUploadArea() {
  return `
    <div class="upload-area" id="uploadArea">
      <div class="upload-icon">📤</div>
      <p class="upload-text">Drop an image here or click to upload</p>
      <p class="upload-hint">Supports JPG, PNG, WebP • Max 10MB</p>
      <input type="file" class="file-input" id="fileInput" accept="image/*" />
    </div>
  `;
}

// Render camera area
function renderCameraArea() {
  return `
    <div class="camera-container" id="cameraContainer">
      <video class="camera-video" id="cameraVideo" autoplay playsinline></video>
      <div class="camera-overlay"></div>
      <div class="camera-frame"></div>
      <div class="scanning-line" id="scanningLine" style="display: none;"></div>
    </div>
    <div class="camera-controls">
      <button class="camera-btn primary" id="captureBtn">
        <span>📸</span> Capture
      </button>
      <button class="camera-btn secondary" id="toggleCameraBtn">
        <span>🔄</span> Switch Camera
      </button>
    </div>
  `;
}

// Render empty results
function renderEmptyResults() {
  return `
    <div class="results-empty">
      <div class="results-empty-icon">𓏤𓏤𓏤</div>
      <p class="results-empty-text">Upload or capture an image of an Egyptian hieroglyph to identify it</p>
    </div>
  `;
}

// Render loading state
function renderLoading() {
  return `
    <div class="loading">
      <div class="loading-spinner"></div>
      <p class="loading-text">Analyzing hieroglyph...</p>
    </div>
  `;
}

// Render results
function renderResults(predictions) {
  if (!predictions || predictions.length === 0) {
    return renderEmptyResults();
  }

  const demoNotice = isModelDemo() ? `
    <div class="notification" style="position: relative; margin-bottom: 16px;">
      ⚠️ Demo mode - Results are simulated. See dataset setup below to enable real classification.
    </div>
  ` : '';

  return demoNotice + `
    <div class="results-list">
      ${predictions.map((pred, idx) => `
        <div class="result-item" data-prediction-idx="${idx}" style="animation-delay: ${idx * 0.1}s">
          <div class="result-glyph">
            ${getHieroglyphSymbol(pred.code)}
          </div>
          <div class="result-info">
            <h3>${pred.name}</h3>
            <div class="result-code">Gardiner Code: ${pred.code}</div>
            <p class="result-description">
              <strong>Meaning:</strong> ${pred.meaning}<br>
              ${pred.phonetic ? `<strong>Phonetic:</strong> ${pred.phonetic}<br>` : ''}
              <strong>Category:</strong> ${pred.category}
            </p>
            <div class="result-confidence">
              <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
              </div>
              <span class="confidence-text">${(pred.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="feedback-buttons" style="display: flex; gap: 8px; margin-top: 12px;">
              <button class="feedback-btn correct" onclick="sendFeedback('${pred.code}', true, ${idx})" style="
                background: rgba(40, 167, 69, 0.2);
                border: 1px solid rgba(40, 167, 69, 0.5);
                color: #28a745;
                padding: 6px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                display: flex;
                align-items: center;
                gap: 4px;
              ">✓ Correct</button>
              <button class="feedback-btn wrong" onclick="sendFeedback('${pred.code}', false, ${idx})" style="
                background: rgba(220, 53, 69, 0.2);
                border: 1px solid rgba(220, 53, 69, 0.5);
                color: #dc3545;
                padding: 6px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                display: flex;
                align-items: center;
                gap: 4px;
              ">✗ Wrong</button>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// Get hieroglyph symbol (uses Unicode Egyptian Hieroglyphs block when available)
function getHieroglyphSymbol(code) {
  // Map some common Gardiner codes to Unicode hieroglyphs
  const unicodeMap = {
    'A1': '𓀀', 'A2': '𓀁', 'A14': '𓀍', 'A17': '𓀐', 'A24': '𓀗', 'A26': '𓀙', 'A28': '𓀛', 'A40': '𓀭',
    'D1': '𓁶', 'D2': '𓁷', 'D4': '𓁹', 'D10': '𓂀', 'D21': '𓂋', 'D28': '𓂓', 'D36': '𓂝', 'D46': '𓂧', 'D54': '𓂻', 'D58': '𓃀',
    'G1': '𓄿', 'G5': '𓅃', 'G7': '𓅆', 'G14': '𓅓', 'G17': '𓅓', 'G29': '𓅡', 'G35': '𓅨', 'G36': '𓅪', 'G37': '𓅫', 'G38': '𓅬', 'G39': '𓅭', 'G43': '𓅱',
    'I9': '𓆑', 'I10': '𓆓',
    'L1': '𓆣', 'L2': '𓆤',
    'M17': '𓇋', 'M23': '𓇓',
    'N1': '𓇯', 'N5': '𓇳', 'N14': '𓇼', 'N35': '𓈖', 'N36': '𓈗',
    'O1': '𓉐', 'O4': '𓉔', 'O34': '𓊃', 'O49': '𓊖',
    'Q3': '𓊪',
    'R4': '𓊵', 'R8': '𓊹', 'R11': '𓋴',
    'S29': '𓋴', 'S34': '𓋹', 'S38': '𓌀', 'S43': '𓌙',
    'T14': '𓌳',
    'V13': '𓍿', 'V28': '𓎛', 'V30': '𓎟', 'V31': '𓎡',
    'W11': '𓏏',
    'X1': '𓏏',
    'Y1': '𓏛',
    'Z1': '𓏤', 'Z4': '𓏭'
  };

  return unicodeMap[code] || `<span style="font-size: 24px; color: #666;">${code}</span>`;
}

// Render dataset information with credentials setup
function renderDatasetInfo() {
  return `
    <div style="display: grid; gap: 24px;">
      <div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 12px;">
        <h3 style="color: #c9a227; margin-bottom: 12px;">✅ Downloaded Datasets (Ready to Use)</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 8px 0;"><strong>EgyptianHieroglyphDataset (Franken)</strong></td>
            <td style="color: #40e0d0;">21,427 images</td>
            <td>datasets/EgyptianHieroglyphDataset/</td>
          </tr>
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 8px 0;"><strong>Glyphnet Manual Dataset</strong></td>
            <td style="color: #40e0d0;">8,433 images</td>
            <td>datasets/Glyphnet/Manual_extracted/</td>
          </tr>
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 8px 0;"><strong>Glyphnet Automated Dataset</strong></td>
            <td style="color: #40e0d0;">8,971 images</td>
            <td>datasets/Glyphnet/Automated_extracted/</td>
          </tr>
          <tr>
            <td style="padding: 8px 0;"><strong>Pre-trained Model Weights</strong></td>
            <td style="color: #40e0d0;">2 MB</td>
            <td>datasets/Glyphnet/weights.hdf5</td>
          </tr>
        </table>
      </div>
      
      <div style="background: rgba(201, 162, 39, 0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(201, 162, 39, 0.3);">
        <h3 style="color: #c9a227; margin-bottom: 12px;">🔐 Kaggle Datasets (Require Authentication)</h3>
        <p style="margin-bottom: 16px; opacity: 0.8;">To download additional datasets from Kaggle, follow these steps:</p>
        
        <ol style="margin-left: 20px; line-height: 1.8;">
          <li>Go to <a href="https://www.kaggle.com/settings" target="_blank" style="color: #40e0d0;">kaggle.com/settings</a></li>
          <li>Scroll to "API" section and click <strong>"Create New Token"</strong></li>
          <li>This downloads <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">kaggle.json</code></li>
          <li>Move it to: <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">~/.kaggle/kaggle.json</code></li>
          <li>Set permissions: <code style="background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px;">chmod 600 ~/.kaggle/kaggle.json</code></li>
          <li>Run these commands to download:
            <pre style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin-top: 8px; overflow-x: auto;">
cd /Users/yousef/.gemini/antigravity/scratch/hieroglyphics-identifier/datasets
kaggle datasets download -d ahmedsamir100/egyptian-hieroglyphs-glyphdataset
kaggle datasets download -d oussamaerrifai/hieroglyphsdataset
unzip -q "*.zip"</pre>
          </li>
        </ol>
        
        <div style="margin-top: 16px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px;">
          <strong>Available Kaggle Datasets:</strong>
          <ul style="margin-left: 20px; margin-top: 8px;">
            <li><a href="https://www.kaggle.com/datasets/ahmedsamir100/egyptian-hieroglyphs-glyphdataset" target="_blank" style="color: #40e0d0;">GlyphDataset</a> - 17,409 files</li>
            <li><a href="https://www.kaggle.com/datasets/oussamaerrifai/hieroglyphsdataset" target="_blank" style="color: #40e0d0;">Hieroglyphs_Dataset</a> - 4,032 images</li>
          </ul>
        </div>
      </div>
      
      <div style="background: rgba(201, 76, 76, 0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(201, 76, 76, 0.3);">
        <h3 style="color: #c94c4c; margin-bottom: 12px;">🔒 Hugging Face Dataset (Requires Account Approval)</h3>
        <p style="margin-bottom: 12px; opacity: 0.8;">The HLA (Hieroglyphic Layout Analysis) dataset requires requesting access:</p>
        
        <ol style="margin-left: 20px; line-height: 1.8;">
          <li>Create a <a href="https://huggingface.co/join" target="_blank" style="color: #40e0d0;">Hugging Face account</a></li>
          <li>Go to <a href="https://huggingface.co/datasets/AhmedElTaher/Egyptian_Hieroglyphic_Layout_Analysis_HLA" target="_blank" style="color: #40e0d0;">HLA Dataset page</a></li>
          <li>Click <strong>"Request Access"</strong> and wait for approval</li>
          <li>Once approved, run:
            <pre style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; margin-top: 8px;">
huggingface-cli login
# Paste your HF token when prompted
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AhmedElTaher/Egyptian_Hieroglyphic_Layout_Analysis_HLA', repo_type='dataset', local_dir='./datasets/HLA_Dataset')"</pre>
          </li>
        </ol>
        
        <p style="margin-top: 12px; font-size: 0.9rem;"><strong>Note:</strong> This dataset contains 897 high-res images for layout analysis and segmentation.</p>
      </div>
      
      <div style="background: rgba(64, 224, 208, 0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(64, 224, 208, 0.3);">
        <h3 style="color: #40e0d0; margin-bottom: 12px;">🚀 Convert Model for Web Use</h3>
        <p style="margin-bottom: 12px; opacity: 0.8;">To enable real classification (instead of demo mode), convert the Glyphnet model:</p>
        
        <pre style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; overflow-x: auto;">
# Install TensorFlow.js converter
pip install tensorflowjs

# Convert the Keras model to TensorFlow.js format
cd /Users/yousef/.gemini/antigravity/scratch/hieroglyphics-identifier/datasets/Glyphnet
tensorflowjs_converter --input_format=keras weights.hdf5 ../tfjs_model/

# Move to app's public folder
mv ../tfjs_model ../app/public/model</pre>
        
        <p style="margin-top: 12px; font-size: 0.9rem;">After conversion, update <code>classifier.js</code> to load the model from <code>/model/model.json</code></p>
      </div>
    </div>
  `;
}

// Attach event listeners
function attachEventListeners() {
  // Mode toggle buttons
  document.getElementById('uploadModeBtn')?.addEventListener('click', () => {
    currentMode = 'upload';
    stopCamera();
    renderApp();
  });

  document.getElementById('cameraModeBtn')?.addEventListener('click', async () => {
    currentMode = 'camera';
    renderApp();
    await startCamera();
  });

  // Upload area
  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('fileInput');

  if (uploadArea && fileInput) {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleImageUpload(files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
      }
    });
  }

  // Camera controls
  document.getElementById('captureBtn')?.addEventListener('click', handleCapture);
  document.getElementById('toggleCameraBtn')?.addEventListener('click', toggleCamera);
}

// Handle image upload
async function handleImageUpload(file) {
  if (!file.type.startsWith('image/')) {
    showNotification('Please upload an image file', 'error');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showNotification('File size must be less than 10MB', 'error');
    return;
  }

  const reader = new FileReader();
  reader.onload = async (e) => {
    const img = new Image();
    img.onload = async () => {
      showPreview(img.src);
      await processImage(img);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// Show image preview
function showPreview(src) {
  const previewArea = document.getElementById('previewArea');
  if (previewArea) {
    previewArea.innerHTML = `
      <div class="enhance-controls" style="display: flex; justify-content: center; margin-bottom: 12px;">
        <button class="enhance-btn ${isEnhanced ? 'active' : ''}" id="enhanceBtn" style="
          background: ${isEnhanced ? 'rgba(255, 193, 7, 0.8)' : 'rgba(255, 193, 7, 0.1)'};
          border: 1px solid rgba(255, 193, 7, 0.3);
          color: ${isEnhanced ? '#1a1a1a' : '#ffc107'};
          padding: 8px 16px;
          border-radius: 20px;
          cursor: pointer;
          font-family: 'Outfit', sans-serif;
          font-size: 0.9rem;
          display: flex;
          align-items: center;
          gap: 8px;
        ">
          <span>🔦</span> Enhance for Stone
        </button>
      </div>
      <div class="preview-container" style="margin-top: 10px; position: relative;">
        <img src="${src}" alt="Uploaded hieroglyph" class="preview-image" id="previewImage" data-original-src="${src}" />
        <button class="clear-btn" id="clearBtn">✕</button>
      </div>
    `;

    document.getElementById('clearBtn')?.addEventListener('click', () => {
      previewArea.innerHTML = '';
      document.getElementById('resultsArea').innerHTML = renderEmptyResults();
      isEnhanced = false;
    });

    // Enhance button - toggle and reprocess
    document.getElementById('enhanceBtn')?.addEventListener('click', async () => {
      isEnhanced = !isEnhanced;
      const btn = document.getElementById('enhanceBtn');
      btn.style.background = isEnhanced ? 'rgba(255, 193, 7, 0.8)' : 'rgba(255, 193, 7, 0.1)';
      btn.style.color = isEnhanced ? '#1a1a1a' : '#ffc107';

      const img = document.getElementById('previewImage');
      const originalSrc = img.getAttribute('data-original-src');

      // Reload original and reprocess
      const tempImg = new Image();
      tempImg.onload = async () => {
        await processImage(tempImg);
      };
      tempImg.src = originalSrc;
    });

    // Hide upload area
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
      uploadArea.style.display = 'none';
    }
  }
}

// ... (imports moved to top)

// ... (existing imports)

// Process image for classification
async function processImage(imageElement) {
  if (isProcessing) return;
  isProcessing = true;

  const resultsArea = document.getElementById('resultsArea');
  if (resultsArea) {
    resultsArea.innerHTML = renderLoading();
  }

  // Ensure detector is loaded
  await loadDetector();

  try {
    // 0. Apply enhancement if enabled (CLAHE + sharpening for stone photos)
    let processedImage = imageElement;
    if (isEnhanced) {
      console.log("Applying stone enhancement (CLAHE + sharpening)...");
      processedImage = enhanceImage(imageElement);

      // Update preview to show enhanced version
      const previewImg = document.getElementById('previewImage');
      if (previewImg && processedImage.toDataURL) {
        previewImg.src = processedImage.toDataURL();
      }
    }

    // 1. Run Object Detection
    console.log("Detecting glyphs...");
    const boxes = await detectGlyphs(processedImage);
    console.log(`Found ${boxes.length} glyphs`);

    let allPredictions = [];

    if (boxes.length > 0) {
      // 2. Draw boxes on preview
      drawBoxesOnPreview(boxes);

      // 3. Classify each detected glyph
      for (const box of boxes) {
        // Crop the glyph
        const crop = cropImage(processedImage, box);

        // Classify the crop
        const glyphResults = await classifyImage(crop);
        const bestMatch = glyphResults[0]; // Top 1 result

        allPredictions.push({
          box,
          ...bestMatch
        });
      }
    } else {
      // Fallback: Classify entire image if no specific glyphs detected
      console.log("No boxes detected, classifying full image");
      const glyphResults = await classifyImage(processedImage);
      allPredictions = glyphResults.map(r => ({ box: null, ...r }));
    }

    if (resultsArea) {
      resultsArea.innerHTML = renderResults(allPredictions);
    }
  } catch (error) {
    console.error('Processing error:', error);
    showNotification('Processing failed: ' + error.message, 'error');
    if (resultsArea) {
      resultsArea.innerHTML = renderEmptyResults();
    }
  } finally {
    isProcessing = false;
  }
}

// Helper: Crop image region
function cropImage(sourceImage, box) {
  const canvas = document.createElement('canvas');
  canvas.width = box.w;
  canvas.height = box.h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(sourceImage, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
  return canvas; // classifier accepts canvas
}

// Helper: Draw boxes overlay
function drawBoxesOnPreview(boxes) {
  const previewImage = document.getElementById('previewImage');
  const previewContainer = previewImage.parentElement;

  // Remove old overlay if any
  const oldOverlay = previewContainer.querySelector('.boxes-overlay');
  if (oldOverlay) oldOverlay.remove();

  // Create overlay container
  const overlay = document.createElement('div');
  overlay.className = 'boxes-overlay';
  overlay.style.position = 'absolute';
  overlay.style.top = '0';
  overlay.style.left = '0';
  overlay.style.width = '100%';
  overlay.style.height = '100%';
  overlay.style.pointerEvents = 'none'; // Let clicks pass through

  // We need to scale boxes to match the *displayed* image size vs natural size
  const displayedWidth = previewImage.clientWidth;
  const displayedHeight = previewImage.clientHeight;
  const naturalWidth = previewImage.naturalWidth;
  const naturalHeight = previewImage.naturalHeight;

  const scaleX = displayedWidth / naturalWidth;
  const scaleY = displayedHeight / naturalHeight;

  // Center the overlay over the image (handling object-fit if necessary, but usually simple img fits)
  // For simplicity, we assume styling keeps image centered or fills. 
  // Better: obtain styling rect.

  boxes.forEach(box => {
    const boxEl = document.createElement('div');
    boxEl.style.position = 'absolute';
    boxEl.style.border = '2px solid #00ff00';
    boxEl.style.left = `${box.x * scaleX}px`;
    boxEl.style.top = `${box.y * scaleY}px`;
    boxEl.style.width = `${box.w * scaleX}px`;
    boxEl.style.height = `${box.h * scaleY}px`;

    // Add confidence label
    const label = document.createElement('span');
    label.innerText = `${(box.conf * 100).toFixed(0)}%`;
    label.style.background = '#00ff00';
    label.style.color = 'black';
    label.style.fontSize = '10px';
    label.style.position = 'absolute';
    label.style.top = '-14px';
    label.style.left = '0';
    label.style.padding = '0 2px';

    boxEl.appendChild(label);
    overlay.appendChild(boxEl);
  });

  previewContainer.style.position = 'relative'; // Ensure positioning context
  previewContainer.appendChild(overlay);
}

// Camera functions
let facingMode = 'environment';

async function startCamera() {
  try {
    const video = document.getElementById('cameraVideo');
    if (!video) return;

    const constraints = {
      video: {
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = cameraStream;

    // Show scanning line animation
    const scanningLine = document.getElementById('scanningLine');
    if (scanningLine) {
      scanningLine.style.display = 'block';
    }
  } catch (error) {
    console.error('Camera error:', error);
    showNotification('Could not access camera. Please check permissions.', 'error');
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

async function toggleCamera() {
  stopCamera();
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  await startCamera();
}

async function handleCapture() {
  const video = document.getElementById('cameraVideo');
  if (!video || !cameraStream) return;

  // Create canvas and capture frame
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  // Create image from canvas
  const img = new Image();
  img.onload = async () => {
    showPreview(canvas.toDataURL('image/jpeg'));
    await processImage(img);
  };
  img.src = canvas.toDataURL('image/jpeg');
}

// Show notification
function showNotification(message, type = 'info') {
  const existing = document.querySelector('.notification');
  if (existing) existing.remove();

  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.textContent = message;
  document.body.appendChild(notification);

  setTimeout(() => notification.remove(), 4000);
}

// Feedback storage for reinforcement learning
let feedbackData = JSON.parse(localStorage.getItem('hieroglyphFeedback') || '[]');

// Handle user feedback on predictions
function sendFeedback(code, isCorrect, idx) {
  const timestamp = new Date().toISOString();
  const imageData = document.getElementById('previewImage')?.src || null;

  const feedback = {
    code,
    isCorrect,
    timestamp,
    imageData: imageData ? imageData.substring(0, 100) + '...' : null // Truncate for storage
  };

  feedbackData.push(feedback);
  localStorage.setItem('hieroglyphFeedback', JSON.stringify(feedbackData));

  // Update button states to show feedback was recorded
  const resultItem = document.querySelector(`[data-prediction-idx="${idx}"]`);
  if (resultItem) {
    const buttons = resultItem.querySelector('.feedback-buttons');
    if (buttons) {
      buttons.innerHTML = `
        <span style="color: ${isCorrect ? '#28a745' : '#dc3545'}; font-size: 0.9rem;">
          ${isCorrect ? '✓ Marked as correct' : '✗ Marked as wrong'} - Thank you!
        </span>
      `;
    }
  }

  // Log for debugging
  console.log('Feedback recorded:', feedback);
  console.log('Total feedback entries:', feedbackData.length);

  showNotification(
    isCorrect ? 'Thanks! Marked as correct.' : 'Thanks! We\'ll improve this.',
    isCorrect ? 'success' : 'info'
  );
}

// Expose sendFeedback globally for onclick handlers
window.sendFeedback = sendFeedback;

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
init();
