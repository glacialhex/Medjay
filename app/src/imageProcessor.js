/**
 * Image processing utilities for enhancing etched hieroglyphs
 */

/**
 * Apply a series of filters to the image canvas
 * @param {HTMLCanvasElement} canvas - The source canvas
 * @param {Object} options - Filter options
 * @returns {HTMLCanvasElement} - A new canvas with filters applied
 */
export function enhanceImage(sourceCanvas, options = { contrast: 0, sharpen: 0, grayscale: false }) {
    const width = sourceCanvas.width;
    const height = sourceCanvas.height;

    // Create working canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(sourceCanvas, 0, 0);

    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data; // reference to the pixel data array

    // 1. Grayscale (Optional, helps remove color noise from stone)
    if (options.grayscale) {
        applyGrayscale(data);
    }

    // 2. Contrast Enhancement
    if (options.contrast !== 0) {
        applyContrast(data, options.contrast);
    }

    ctx.putImageData(imageData, 0, 0);

    // 3. Sharpening (Convolution requires neighboring pixels, easier on context)
    if (options.sharpen > 0) {
        // We use a separate pass for convolution
        return applySharpen(canvas, options.sharpen);
    }

    return canvas;
}

function applyGrayscale(data) {
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg;
        data[i + 1] = avg;
        data[i + 2] = avg;
    }
}

function applyContrast(data, contrast) {
    // contrast is -100 to 100.
    // Factor calculation: (259 * (contrast + 255)) / (255 * (259 - contrast))
    // Normalized to usually 0-2 range or similar logic
    const factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

    for (let i = 0; i < data.length; i += 4) {
        data[i] = truncate(factor * (data[i] - 128) + 128);
        data[i + 1] = truncate(factor * (data[i + 1] - 128) + 128);
        data[i + 2] = truncate(factor * (data[i + 2] - 128) + 128);
    }
}

function truncate(value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return value;
}

function applySharpen(canvas, amount) {
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext('2d');
    const inputData = ctx.getImageData(0, 0, w, h);

    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = w;
    outputCanvas.height = h;
    const outCtx = outputCanvas.getContext('2d');
    const outputData = outCtx.createImageData(w, h);

    // Simple 3x3 Sharpen Kernel
    // [  0, -1,  0 ]
    // [ -1,  5, -1 ]
    // [  0, -1,  0 ]
    // Amount scales the center weight and subtracts neighbors

    // A more adjustable kernel based on 'amount' (0.0 to 1.0+)
    const x = amount;
    // Standard sharpen kernel is often: 0,-1,0, -1,5,-1, 0,-1,0 (sum=1)
    // Let's parameterize: center = val, neighbors = -x
    // To maintain brightness, sum of weights should be 1.
    // 4*(-x) + center = 1  => center = 1 + 4x

    const kernel = [
        0, -x, 0,
        -x, 1 + 4 * x, -x,
        0, -x, 0
    ];

    convolve(inputData, outputData, kernel);
    outCtx.putImageData(outputData, 0, 0);
    return outputCanvas;
}

function convolve(src, dst, kernel) {
    const side = Math.round(Math.sqrt(kernel.length));
    const halfSide = Math.floor(side / 2);
    const srcData = src.data;
    const dstData = dst.data;
    const w = src.width;
    const h = src.height;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const dstOff = (y * w + x) * 4;

            let r = 0, g = 0, b = 0;

            for (let cy = 0; cy < side; cy++) {
                for (let cx = 0; cx < side; cx++) {
                    const scy = y + cy - halfSide;
                    const scx = x + cx - halfSide;

                    if (scy >= 0 && scy < h && scx >= 0 && scx < w) {
                        const srcOff = (scy * w + scx) * 4;
                        const wt = kernel[cy * side + cx];
                        r += srcData[srcOff] * wt;
                        g += srcData[srcOff + 1] * wt;
                        b += srcData[srcOff + 2] * wt;
                    }
                }
            }

            dstData[dstOff] = r;
            dstData[dstOff + 1] = g;
            dstData[dstOff + 2] = b;
            dstData[dstOff + 3] = srcData[dstOff + 3]; // Alpha copy
        }
    }
}
