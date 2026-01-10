import * as ort from 'onnxruntime-web';
import { gardinerSigns } from './gardinerSigns';
import { env } from 'onnxruntime-web';

// Configure ONNX Runtime to use WASM backend
env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';

// Model state
let session = null;
let classLabels = [];
let isModelLoading = false;
let modelLoadPromise = null;

/**
 * Load the hieroglyph classification model (ONNX)
 */
export async function loadModel() {
    if (session) return session;
    if (modelLoadPromise) return modelLoadPromise;

    isModelLoading = true;

    modelLoadPromise = new Promise(async (resolve, reject) => {
        try {
            console.log('Loading ONNX model...');

            // Load labels first
            const labelsResponse = await fetch('/model_labels.json');
            if (!labelsResponse.ok) throw new Error('Failed to load labels');
            classLabels = await labelsResponse.json();

            // Create inference session
            // Note: model must be in public folder
            session = await ort.InferenceSession.create('/hieroglyph_model.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            console.log('Hieroglyph classifier ready (ONNX)');
            resolve(session);
        } catch (error) {
            console.error('Failed to load model:', error);
            reject(error);
        } finally {
            isModelLoading = false;
        }
    });

    return modelLoadPromise;
}

/**
 * Preprocess image for the model
 * Expects 100x100 RGB image
 * Returns Float32Array [1, 3, 100, 100] normalized 0-1
 */
function preprocessImage(imageElement) {
    const width = 100;
    const height = 100;

    // Create a temporary canvas to resize and get pixel data
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Draw and resize
    ctx.drawImage(imageElement, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const { data } = imageData;

    // Convert to float32 and rearrange to NCHW [1, 3, 100, 100]
    const float32Data = new Float32Array(1 * 3 * width * height);

    for (let i = 0; i < width * height; i++) {
        // Normalize 0-255 -> 0.0-1.0
        const r = data[i * 4] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;

        // Fill tensor data (Planar format: RRR...GGG...BBB...)
        float32Data[i] = r;
        float32Data[i + (width * height)] = g;
        float32Data[i + (width * height * 2)] = b;
    }

    return new ort.Tensor('float32', float32Data, [1, 3, width, height]);
}

/**
 * Softmax function
 */
function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

/**
 * Classify a hieroglyph image
 */
export async function classifyImage(imageElement) {
    const loadedSession = await loadModel();

    if (!loadedSession) {
        throw new Error('Model not available');
    }

    try {
        // Preprocess
        const tensor = preprocessImage(imageElement);

        // Run inference
        const feeds = { input: tensor }; // 'input' matches the ONNX export input name
        const results = await loadedSession.run(feeds);

        // Get output
        const output = results.output; // 'output' matches the ONNX export output name
        const probabilities = softmax(Array.from(output.data));

        // Get top 5 predictions
        const predictions = probabilities
            .map((prob, idx) => ({ classIndex: idx, confidence: prob }))
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 5);

        // Map predictions to Gardiner signs
        return predictions.map(pred => {
            const code = classLabels[pred.classIndex] || 'Unknown';
            const signInfo = gardinerSigns[code] || {
                name: 'Unknown Sign',
                meaning: 'Not in database',
                phonetic: '',
                category: 'Unknown'
            };

            return {
                code,
                confidence: pred.confidence,
                ...signInfo
            };
        });
    } catch (e) {
        console.error("Inference failed", e);
        alert(`Inference failed: ${e.message}`);
        throw e;
    }
}

/**
 * Get model status
 */
export function getModelStatus() {
    if (session) return 'ready';
    if (isModelLoading) return 'loading';
    return 'not_loaded';
}

/**
 * Check if model is in demo mode
 */
export function isModelDemo() {
    return false;
}
