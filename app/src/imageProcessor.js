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
