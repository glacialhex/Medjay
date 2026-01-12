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
 * Preprocess image for the ResNet18 model
 * Expects 224x224 RGB image with ImageNet normalization
 * Returns Float32Array [1, 3, 224, 224] 
 */
function preprocessImage(imageElement) {
    const width = 224;
    const height = 224;

    // Create a temporary canvas to resize and get pixel data
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Draw and resize
    ctx.drawImage(imageElement, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const { data } = imageData;

    // ImageNet normalization values
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Convert to float32 and rearrange to NCHW [1, 3, 224, 224]
    const float32Data = new Float32Array(1 * 3 * width * height);

    for (let i = 0; i < width * height; i++) {
        // Normalize 0-255 -> 0.0-1.0, then apply ImageNet normalization
        const r = (data[i * 4] / 255.0 - mean[0]) / std[0];
        const g = (data[i * 4 + 1] / 255.0 - mean[1]) / std[1];
        const b = (data[i * 4 + 2] / 255.0 - mean[2]) / std[2];

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
        let probabilities = softmax(Array.from(output.data));

        // Apply pseudo-reinforcement: adjust probabilities based on correction history
        probabilities = applyReinforcementLearning(probabilities);

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
 * Apply pseudo-reinforcement learning based on user corrections
 * Adjusts probabilities to favor correct codes and penalize wrong ones
 */
function applyReinforcementLearning(probabilities) {
    // Load correction history from localStorage
    const corrections = JSON.parse(localStorage.getItem('hieroglyphCorrections') || '[]');
    const confirmations = JSON.parse(localStorage.getItem('hieroglyphFeedback') || '[]');

    if (corrections.length === 0 && confirmations.length === 0) {
        return probabilities; // No history, return unchanged
    }

    // Build adjustment map: code -> adjustment factor
    const adjustments = {};

    // Learn from corrections: boost correct codes, penalize wrong ones
    corrections.forEach(c => {
        if (c.correctCode && c.correctCode !== 'unknown') {
            // Boost the correct answer
            const correctIdx = classLabels.indexOf(c.correctCode);
            if (correctIdx !== -1) {
                adjustments[correctIdx] = (adjustments[correctIdx] || 1) * 1.3; // 30% boost per correction
            }
        }
        if (c.predictedCode) {
            // Penalize the wrong prediction
            const wrongIdx = classLabels.indexOf(c.predictedCode);
            if (wrongIdx !== -1) {
                adjustments[wrongIdx] = (adjustments[wrongIdx] || 1) * 0.7; // 30% penalty per correction
            }
        }
    });

    // Learn from confirmations: boost confirmed codes
    confirmations.forEach(c => {
        if (c.isCorrect && c.predictedCode) {
            const confirmedIdx = classLabels.indexOf(c.predictedCode);
            if (confirmedIdx !== -1) {
                adjustments[confirmedIdx] = (adjustments[confirmedIdx] || 1) * 1.1; // 10% boost per confirmation
            }
        }
    });

    // Apply adjustments
    const adjusted = probabilities.map((p, idx) => {
        const factor = adjustments[idx] || 1;
        return p * factor;
    });

    // Re-normalize to sum to 1
    const sum = adjusted.reduce((a, b) => a + b, 0);
    const normalized = adjusted.map(p => p / sum);

    console.log('Applied reinforcement learning:', Object.keys(adjustments).length, 'adjustments');

    return normalized;
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
