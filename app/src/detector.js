import * as ort from 'onnxruntime-web';
import { env } from 'onnxruntime-web';

// Configure ONNX Runtime to use WASM backend (shared with classifier)
env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';

let detectorSession = null;
let isDetectorLoading = false;
let detectorLoadPromise = null;

// YOLOv8 Config
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

/**
 * Load the YOLO object detection model
 */
export async function loadDetector() {
    if (detectorSession) return detectorSession;
    if (detectorLoadPromise) return detectorLoadPromise;

    isDetectorLoading = true;
    detectorLoadPromise = new Promise(async (resolve, reject) => {
        try {
            console.log('Loading YOLO detector...');
            detectorSession = await ort.InferenceSession.create('/yolov8_model.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            console.log('YOLO detector ready');
            resolve(detectorSession);
        } catch (e) {
            console.error('Failed to load detector:', e);
            reject(e);
        } finally {
            isDetectorLoading = false;
        }
    });
    return detectorLoadPromise;
}

/**
 * Detect glyphs in an image
 */
export async function detectGlyphs(imageElement) {
    const session = await loadDetector();
    if (!session) throw new Error('Detector not loaded');

    // 1. Preprocess: Resize to 640x640 (preserving aspect ratio, padding)
    const { tensor, scale, xPadding, yPadding } = preprocess(imageElement);

    // 2. Inference
    const results = await session.run({ images: tensor });
    const output = results.output0; // YOLOv8 output name is usually 'output0'

    // 3. Post-process: Parse boxes and NMS
    const boxes = postprocess(output, scale, xPadding, yPadding, imageElement.width, imageElement.height);

    return boxes;
}

function preprocess(image) {
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d');

    // Fill gray background
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

    // Calculate scale to fit
    const scale = Math.min(INPUT_SIZE / image.width, INPUT_SIZE / image.height);
    const newWidth = image.width * scale;
    const newHeight = image.height * scale;
    const xPadding = (INPUT_SIZE - newWidth) / 2;
    const yPadding = (INPUT_SIZE - newHeight) / 2;

    // Draw scaled image
    ctx.drawImage(image, xPadding, yPadding, newWidth, newHeight);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = imageData;
    const float32Data = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        float32Data[i] = data[i * 4] / 255.0; // R
        float32Data[i + INPUT_SIZE * INPUT_SIZE] = data[i * 4 + 1] / 255.0; // G
        float32Data[i + INPUT_SIZE * INPUT_SIZE * 2] = data[i * 4 + 2] / 255.0; // B
    }

    return {
        tensor: new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]),
        scale,
        xPadding,
        yPadding
    };
}

function postprocess(output, scale, xPadding, yPadding, originalW, originalH) {
    // YOLOv8 output: [1, 5, 8400] -> 5 rows: xc, yc, w, h, conf
    const data = output.data;
    const numBoxes = 8400; // 640/8=80, 640/16=40, 640/32=20 -> 6400+1600+400=8400
    const boxes = [];

    // Extract potential boxes
    for (let i = 0; i < numBoxes; i++) {
        // Data layout is [1, 5, 8400]. 
        // Index access: row * numBoxes + col
        // Rows: 0=xc, 1=yc, 2=w, 3=h, 4=conf (for 1 class)

        const conf = data[4 * numBoxes + i];

        if (conf > CONF_THRESHOLD) {
            const xc = data[0 * numBoxes + i];
            const yc = data[1 * numBoxes + i];
            const w = data[2 * numBoxes + i];
            const h = data[3 * numBoxes + i];

            // Convert to xyxy (un-padded coords)
            let x1 = (xc - w / 2 - xPadding) / scale;
            let y1 = (yc - h / 2 - yPadding) / scale;
            let x2 = (xc + w / 2 - xPadding) / scale;
            let y2 = (yc + h / 2 - yPadding) / scale;

            // Clip
            x1 = Math.max(0, x1);
            y1 = Math.max(0, y1);
            x2 = Math.min(originalW, x2);
            y2 = Math.min(originalH, y2);

            boxes.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1, conf });
        }
    }

    return nms(boxes);
}

// Simple NMS
function nms(boxes) {
    if (boxes.length === 0) return [];

    // Sort by confidence
    boxes.sort((a, b) => b.conf - a.conf);

    const selected = [];
    const active = new Array(boxes.length).fill(true);

    for (let i = 0; i < boxes.length; i++) {
        if (!active[i]) continue;

        selected.push(boxes[i]);

        for (let j = i + 1; j < boxes.length; j++) {
            if (active[j]) {
                const iou = calculateIoU(boxes[i], boxes[j]);
                if (iou > IOU_THRESHOLD) {
                    active[j] = false;
                }
            }
        }
    }
    return selected;
}

function calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);

    if (x2 < x1 || y2 < y1) return 0;

    const intersection = (x2 - x1) * (y2 - y1);
    const areaA = a.w * a.h;
    const areaB = b.w * b.h;

    return intersection / (areaA + areaB - intersection);
}
