/**
 * Image processing utilities for enhancing etched hieroglyphs
 * Applies CLAHE-like contrast enhancement and sharpening to make
 * faint stone carvings more visible to the AI
 */

/**
 * Apply enhancement filters to make hieroglyphs pop from stone background
 * @param {HTMLCanvasElement|HTMLImageElement} source - The source image
 * @returns {HTMLCanvasElement} - Enhanced canvas
 */
export function enhanceImage(source) {
    // Create canvas from source
    const canvas = document.createElement('canvas');
    canvas.width = source.width || source.naturalWidth;
    canvas.height = source.height || source.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // 1. Convert to grayscale for processing (helps with stone)
    const gray = new Uint8Array(data.length / 4);
    for (let i = 0; i < data.length; i += 4) {
        gray[i / 4] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    // 2. Apply CLAHE-like local contrast enhancement
    const enhanced = applyCLAHE(gray, canvas.width, canvas.height);

    // 3. Apply to image data (keep as grayscale - hieroglyphs don't need color)
    for (let i = 0; i < enhanced.length; i++) {
        const idx = i * 4;
        data[idx] = enhanced[i];
        data[idx + 1] = enhanced[i];
        data[idx + 2] = enhanced[i];
    }

    ctx.putImageData(imageData, 0, 0);

    // 4. Apply sharpening
    return applySharpen(canvas, 0.5);
}

/**
 * Simplified CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * Divides image into tiles and equalizes each locally
 */
function applyCLAHE(gray, width, height) {
    const tileSize = 32; // Tile size for local processing
    const output = new Uint8Array(gray.length);

    // Process each tile
    for (let ty = 0; ty < height; ty += tileSize) {
        for (let tx = 0; tx < width; tx += tileSize) {
            const tileW = Math.min(tileSize, width - tx);
            const tileH = Math.min(tileSize, height - ty);

            // Build histogram for this tile
            const hist = new Array(256).fill(0);
            for (let y = ty; y < ty + tileH; y++) {
                for (let x = tx; x < tx + tileW; x++) {
                    hist[gray[y * width + x]]++;
                }
            }

            // Clip histogram (limit contrast)
            const clipLimit = Math.max(1, Math.floor((tileW * tileH) / 256 * 3));
            let excess = 0;
            for (let i = 0; i < 256; i++) {
                if (hist[i] > clipLimit) {
                    excess += hist[i] - clipLimit;
                    hist[i] = clipLimit;
                }
            }

            // Redistribute excess
            const increment = Math.floor(excess / 256);
            for (let i = 0; i < 256; i++) {
                hist[i] += increment;
            }

            // Build CDF (cumulative distribution function)
            const cdf = new Array(256);
            cdf[0] = hist[0];
            for (let i = 1; i < 256; i++) {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            // Normalize CDF to 0-255
            const cdfMin = cdf.find(v => v > 0) || 0;
            const cdfMax = cdf[255];
            const scale = 255 / (cdfMax - cdfMin || 1);

            // Apply equalization to tile
            for (let y = ty; y < ty + tileH; y++) {
                for (let x = tx; x < tx + tileW; x++) {
                    const idx = y * width + x;
                    output[idx] = Math.round((cdf[gray[idx]] - cdfMin) * scale);
                }
            }
        }
    }

    return output;
}

/**
 * Apply unsharp mask sharpening
 */
function applySharpen(canvas, amount) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const src = ctx.getImageData(0, 0, w, h);
    const dst = ctx.createImageData(w, h);

    // Sharpen kernel: center = 1 + 4*amount, neighbors = -amount
    const kernel = [
        0, -amount, 0,
        -amount, 1 + 4 * amount, -amount,
        0, -amount, 0
    ];

    const srcData = src.data;
    const dstData = dst.data;

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            for (let c = 0; c < 3; c++) {
                let sum = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = ((y + ky) * w + (x + kx)) * 4 + c;
                        const ki = (ky + 1) * 3 + (kx + 1);
                        sum += srcData[idx] * kernel[ki];
                    }
                }
                const dstIdx = (y * w + x) * 4 + c;
                dstData[dstIdx] = Math.max(0, Math.min(255, sum));
            }
            dstData[(y * w + x) * 4 + 3] = 255; // Alpha
        }
    }

    // Copy edges
    for (let x = 0; x < w; x++) {
        for (let c = 0; c < 4; c++) {
            dstData[x * 4 + c] = srcData[x * 4 + c];
            dstData[((h - 1) * w + x) * 4 + c] = srcData[((h - 1) * w + x) * 4 + c];
        }
    }
    for (let y = 0; y < h; y++) {
        for (let c = 0; c < 4; c++) {
            dstData[(y * w) * 4 + c] = srcData[(y * w) * 4 + c];
            dstData[(y * w + w - 1) * 4 + c] = srcData[(y * w + w - 1) * 4 + c];
        }
    }

    ctx.putImageData(dst, 0, 0);
    return canvas;
}

/**
 * Extract edges from image - makes it shadow-invariant
 * Converts stone photos into clean edge maps similar to training data
 * @param {HTMLCanvasElement|HTMLImageElement} source - The source image
 * @returns {HTMLCanvasElement} - Edge map canvas
 */
export function extractEdges(source) {
    const canvas = document.createElement('canvas');
    canvas.width = source.width || source.naturalWidth;
    canvas.height = source.height || source.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0);

    const w = canvas.width;
    const h = canvas.height;
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    // 1. Convert to grayscale
    const gray = new Float32Array(w * h);
    for (let i = 0; i < data.length; i += 4) {
        gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    // 2. Apply Gaussian blur (3x3) to reduce noise
    const blurred = new Float32Array(w * h);
    const gaussKernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const gaussSum = 16;

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            let sum = 0;
            let ki = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    sum += gray[(y + ky) * w + (x + kx)] * gaussKernel[ki++];
                }
            }
            blurred[y * w + x] = sum / gaussSum;
        }
    }

    // 3. Sobel edge detection (gradient magnitude)
    const edges = new Float32Array(w * h);
    let maxEdge = 0;

    // Sobel kernels
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            let gx = 0, gy = 0;
            let ki = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const val = blurred[(y + ky) * w + (x + kx)];
                    gx += val * sobelX[ki];
                    gy += val * sobelY[ki];
                    ki++;
                }
            }
            const magnitude = Math.sqrt(gx * gx + gy * gy);
            edges[y * w + x] = magnitude;
            if (magnitude > maxEdge) maxEdge = magnitude;
        }
    }

    // 4. Normalize and threshold edges
    // Use adaptive threshold based on mean edge strength
    let edgeSum = 0;
    for (let i = 0; i < edges.length; i++) edgeSum += edges[i];
    const meanEdge = edgeSum / edges.length;
    const threshold = meanEdge * 1.5; // Edges above 1.5x mean are significant

    // 5. Create output: dark glyphs on light background (like training data)
    for (let i = 0; i < edges.length; i++) {
        const normalized = maxEdge > 0 ? (edges[i] / maxEdge) * 255 : 0;
        // Invert: strong edges become dark (like hieroglyph strokes)
        const value = edges[i] > threshold ? 50 : 200; // Binary-ish with some gradient
        const idx = i * 4;
        data[idx] = value;
        data[idx + 1] = value;
        data[idx + 2] = value;
        data[idx + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);

    // 6. Apply morphological closing (dilate then erode) to connect edges
    return morphClose(canvas, 1);
}

/**
 * Morphological close operation (dilate then erode)
 * Helps connect broken edges from carved hieroglyphs
 */
function morphClose(canvas, radius) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const src = ctx.getImageData(0, 0, w, h);
    const dilated = ctx.createImageData(w, h);
    const closed = ctx.createImageData(w, h);

    // Dilate (take minimum - since our edges are dark)
    for (let y = radius; y < h - radius; y++) {
        for (let x = radius; x < w - radius; x++) {
            let minVal = 255;
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const idx = ((y + ky) * w + (x + kx)) * 4;
                    if (src.data[idx] < minVal) minVal = src.data[idx];
                }
            }
            const dstIdx = (y * w + x) * 4;
            dilated.data[dstIdx] = minVal;
            dilated.data[dstIdx + 1] = minVal;
            dilated.data[dstIdx + 2] = minVal;
            dilated.data[dstIdx + 3] = 255;
        }
    }

    // Erode (take maximum)
    for (let y = radius; y < h - radius; y++) {
        for (let x = radius; x < w - radius; x++) {
            let maxVal = 0;
            for (let ky = -radius; ky <= radius; ky++) {
                for (let kx = -radius; kx <= radius; kx++) {
                    const idx = ((y + ky) * w + (x + kx)) * 4;
                    if (dilated.data[idx] > maxVal) maxVal = dilated.data[idx];
                }
            }
            const dstIdx = (y * w + x) * 4;
            closed.data[dstIdx] = maxVal;
            closed.data[dstIdx + 1] = maxVal;
            closed.data[dstIdx + 2] = maxVal;
            closed.data[dstIdx + 3] = 255;
        }
    }

    ctx.putImageData(closed, 0, 0);
    return canvas;
}
