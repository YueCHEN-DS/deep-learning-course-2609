/**
 * Handwritten Digit Recognition - Multi-Model Course Project
 * Supports 3 Architectures:
 *   1. PyTorch 2-stage CNN (~97k params, 99.11% accuracy)
 *   2. PyTorch Deep MLP (256-64-10, ~218k params, 97.68% accuracy)
 *   3. Scikit-Learn Multinomial Logistic Regression (~7.8k params, 92.59% accuracy)
 *   4. Side-by-Side Comparison Mode (All 3 models on the same drawing)
 */

// --- 1. PyTorch CNN Engine ---
class CNNInferenceEngine {
  constructor(weightsData) {
    this.weights = {};
    this.ready = false;
    if (weightsData) this.loadWeights(weightsData);
  }

  loadWeights(weightsData) {
    for (const key of Object.keys(weightsData)) {
      this.weights[key] = this.decodeBase64Float32(weightsData[key].b64);
    }
    this.ready = true;
  }

  decodeBase64Float32(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  conv2d(input, Cin, H, W, weights, bias, Cout, KH = 3, KW = 3, pad = 1) {
    const outH = H;
    const outW = W;
    const output = new Float32Array(Cout * outH * outW);

    for (let co = 0; co < Cout; co++) {
      const b = bias[co];
      const wCoOffset = co * Cin * KH * KW;
      const outCoOffset = co * outH * outW;

      for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
          let sum = b;
          for (let ci = 0; ci < Cin; ci++) {
            const inCiOffset = ci * H * W;
            const wCiOffset = wCoOffset + ci * KH * KW;
            for (let ky = 0; ky < KH; ky++) {
              const iy = y + ky - pad;
              if (iy < 0 || iy >= H) continue;
              for (let kx = 0; kx < KW; kx++) {
                const ix = x + kx - pad;
                if (ix < 0 || ix >= W) continue;
                sum += input[inCiOffset + iy * W + ix] * weights[wCiOffset + ky * KW + kx];
              }
            }
          }
          output[outCoOffset + y * outW + x] = sum > 0 ? sum : 0;
        }
      }
    }
    return { data: output, H: outH, W: outW };
  }

  maxpool2d(input, C, H, W) {
    const outH = Math.floor(H / 2);
    const outW = Math.floor(W / 2);
    const output = new Float32Array(C * outH * outW);

    for (let c = 0; c < C; c++) {
      const inCOffset = c * H * W;
      const outCOffset = c * outH * outW;
      for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
          const iy = y * 2;
          const ix = x * 2;
          let maxVal = -Infinity;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const val = input[inCOffset + (iy + dy) * W + (ix + dx)];
              if (val > maxVal) maxVal = val;
            }
          }
          output[outCOffset + y * outW + x] = maxVal;
        }
      }
    }
    return { data: output, H: outH, W: outW };
  }

  linear(input, weights, bias, inF, outF, relu = false) {
    const output = new Float32Array(outF);
    for (let o = 0; o < outF; o++) {
      let sum = bias[o];
      const wOffset = o * inF;
      for (let i = 0; i < inF; i++) {
        sum += input[i] * weights[wOffset + i];
      }
      output[o] = relu ? (sum > 0 ? sum : 0) : sum;
    }
    return output;
  }

  softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    let sum = 0;
    const exp = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      exp[i] = Math.exp(arr[i] - max);
      sum += exp[i];
    }
    for (let i = 0; i < arr.length; i++) exp[i] /= sum;
    return exp;
  }

  predict(inputFlat28x28) {
    if (!this.ready) throw new Error('CNN weights not loaded');
    const c1 = this.conv2d(inputFlat28x28, 1, 28, 28, this.weights['conv1.weight'], this.weights['conv1.bias'], 16);
    const p1 = this.maxpool2d(c1.data, 16, c1.H, c1.W);
    const c2 = this.conv2d(p1.data, 16, p1.H, p1.W, this.weights['conv2.weight'], this.weights['conv2.bias'], 32);
    const p2 = this.maxpool2d(c2.data, 32, c2.H, c2.W);
    const fc1 = this.linear(p2.data, this.weights['fc1.weight'], this.weights['fc1.bias'], 1568, 96, true);
    const fc2 = this.linear(fc1, this.weights['fc2.weight'], this.weights['fc2.bias'], 96, 10, false);
    return this.softmax(fc2);
  }
}

// --- 2. PyTorch MLP Engine (Dense Feed-Forward) ---
class MLPInferenceEngine {
  constructor(weightsData) {
    this.weights = {};
    this.ready = false;
    if (weightsData) this.loadWeights(weightsData);
  }

  loadWeights(weightsData) {
    for (const key of Object.keys(weightsData)) {
      this.weights[key] = this.decodeBase64Float32(weightsData[key].b64);
    }
    this.ready = true;
  }

  decodeBase64Float32(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  linear(input, weights, bias, inF, outF, relu = false) {
    const output = new Float32Array(outF);
    for (let o = 0; o < outF; o++) {
      let sum = bias[o];
      const wOffset = o * inF;
      for (let i = 0; i < inF; i++) {
        sum += input[i] * weights[wOffset + i];
      }
      output[o] = relu ? (sum > 0 ? sum : 0) : sum;
    }
    return output;
  }

  softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    let sum = 0;
    const exp = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      exp[i] = Math.exp(arr[i] - max);
      sum += exp[i];
    }
    for (let i = 0; i < arr.length; i++) exp[i] /= sum;
    return exp;
  }

  predict(inputFlat784) {
    if (!this.ready) throw new Error('MLP weights not loaded');
    // Layer 1: 784 -> 256 + ReLU
    const h1 = this.linear(inputFlat784, this.weights['fc1.weight'], this.weights['fc1.bias'], 784, 256, true);
    // Layer 2: 256 -> 64 + ReLU
    const h2 = this.linear(h1, this.weights['fc2.weight'], this.weights['fc2.bias'], 256, 64, true);
    // Layer 3: 64 -> 10 Logits
    const logits = this.linear(h2, this.weights['fc3.weight'], this.weights['fc3.bias'], 64, 10, false);
    return this.softmax(logits);
  }
}

// --- 3. Scikit-Learn Linear Engine ---
class SklearnLinearEngine {
  constructor(weightsData) {
    this.coef = null;
    this.intercept = null;
    this.ready = false;
    if (weightsData) this.loadWeights(weightsData);
  }

  loadWeights(weightsData) {
    this.coef = this.decodeBase64Float32(weightsData.coef.b64);
    this.intercept = this.decodeBase64Float32(weightsData.intercept.b64);
    this.ready = true;
  }

  decodeBase64Float32(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  softmax(arr) {
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    let sum = 0;
    const exp = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) {
      exp[i] = Math.exp(arr[i] - max);
      sum += exp[i];
    }
    for (let i = 0; i < arr.length; i++) exp[i] /= sum;
    return exp;
  }

  predict(inputFlat784) {
    if (!this.ready) throw new Error('Scikit-Learn weights not loaded');
    const numClasses = 10;
    const numFeatures = 784;
    const logits = new Float32Array(numClasses);

    for (let c = 0; c < numClasses; c++) {
      let sum = this.intercept[c];
      const offset = c * numFeatures;
      for (let i = 0; i < numFeatures; i++) {
        sum += inputFlat784[i] * this.coef[offset + i];
      }
      logits[c] = sum;
    }
    return this.softmax(logits);
  }
}

// --- 4. MNIST Preprocessor ---
class MNISTPreprocessor {
  static process(canvas, previewCanvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { width, height } = canvas;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    let totalMass = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const brightness = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        if (brightness > 60) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          totalMass += brightness;
        }
      }
    }

    if (maxX === -1 || totalMass < 1000) return null;

    const pad = 12;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(width - 1, maxX + pad);
    maxY = Math.min(height - 1, maxY + pad);

    const cropW = Math.max(1, maxX - minX + 1);
    const cropH = Math.max(1, maxY - minY + 1);

    const maxDim = Math.max(cropW, cropH);
    const scale = 20 / maxDim;
    const targetW = Math.max(1, Math.round(cropW * scale));
    const targetH = Math.max(1, Math.round(cropH * scale));

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetW;
    tempCanvas.height = targetH;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, minX, minY, cropW, cropH, 0, 0, targetW, targetH);

    const scaledImgData = tempCtx.getImageData(0, 0, targetW, targetH);
    const scaledData = scaledImgData.data;

    let sumX = 0;
    let sumY = 0;
    let scaledMass = 0;

    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < targetW; x++) {
        const idx = (y * targetW + x) * 4;
        const b = Math.max(0, (scaledData[idx] + scaledData[idx + 1] + scaledData[idx + 2]) / 3 - 30);
        if (b > 0) {
          sumX += x * b;
          sumY += y * b;
          scaledMass += b;
        }
      }
    }

    const cX = scaledMass > 0 ? sumX / scaledMass : targetW / 2;
    const cY = scaledMass > 0 ? sumY / scaledMass : targetH / 2;

    const finalCanvas = previewCanvas || document.createElement('canvas');
    finalCanvas.width = 28;
    finalCanvas.height = 28;
    const finalCtx = finalCanvas.getContext('2d');

    finalCtx.fillStyle = '#000000';
    finalCtx.fillRect(0, 0, 28, 28);

    const destX = Math.round(14 - cX);
    const destY = Math.round(14 - cY);

    finalCtx.drawImage(tempCanvas, destX, destY);

    const finalImgData = finalCtx.getImageData(0, 0, 28, 28);
    const finalData = finalImgData.data;
    const normalized = new Float32Array(28 * 28);

    for (let i = 0; i < 28 * 28; i++) {
      const idx = i * 4;
      const b = (finalData[idx] + finalData[idx + 1] + finalData[idx + 2]) / 3;
      normalized[i] = Math.min(1.0, Math.max(0.0, b / 255.0));
    }

    return normalized;
  }
}

// --- 5. UI Controller ---
class DigitApp {
  constructor() {
    this.canvas = document.getElementById('drawing-canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.previewCanvas = document.getElementById('preview-canvas');
    this.canvasHint = document.getElementById('canvas-hint');

    this.clearBtn = document.getElementById('clear-btn');
    this.predictBtn = document.getElementById('predict-btn');

    this.singlePredCard = document.getElementById('single-pred-card');
    this.predictionDisplay = document.getElementById('prediction-value');
    this.confidenceDisplay = document.querySelector('#confidence-value span');
    this.activeModelTag = document.getElementById('active-model-tag');

    // 3-Card Comparison Elements
    this.comparisonCards = document.getElementById('comparison-cards');
    this.compareCnnDigit = document.getElementById('compare-cnn-digit');
    this.compareCnnConf = document.getElementById('compare-cnn-conf');
    this.compareMlpDigit = document.getElementById('compare-mlp-digit');
    this.compareMlpConf = document.getElementById('compare-mlp-conf');
    this.compareSklearnDigit = document.getElementById('compare-sklearn-digit');
    this.compareSklearnConf = document.getElementById('compare-sklearn-conf');

    this.probTitle = document.getElementById('prob-title');
    this.probBarsContainer = document.getElementById('prob-bars');
    this.footerDesc = document.getElementById('footer-desc');

    this.isDrawing = false;
    this.hasDrawn = false;
    this.lastX = 0;
    this.lastY = 0;

    this.currentModel = 'cnn'; // 'cnn' | 'mlp' | 'sklearn' | 'compare'

    // Initialize all 3 ML engines
    this.cnn = new CNNInferenceEngine(window.MODEL_WEIGHTS);
    this.mlp = new MLPInferenceEngine(window.MLP_WEIGHTS);
    this.sklearn = new SklearnLinearEngine(window.SKLEARN_WEIGHTS);

    this.initCanvas();
    this.initEvents();
    this.initProbBars();
    this.updateModelUI();
  }

  initCanvas() {
    this.ctx.fillStyle = '#1c1f24';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    this.ctx.lineWidth = 20;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = '#ffffff';

    const pCtx = this.previewCanvas.getContext('2d');
    pCtx.fillStyle = '#000000';
    pCtx.fillRect(0, 0, 28, 28);
  }

  initProbBars() {
    this.probBarsContainer.innerHTML = '';
    for (let i = 0; i < 10; i++) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.id = `prob-row-${i}`;
      row.innerHTML = `
        <span class="prob-label">${i}</span>
        <div class="prob-track">
          <div class="prob-fill" id="prob-fill-${i}" style="width: 0%"></div>
        </div>
        <span class="prob-val" id="prob-val-${i}">0.0%</span>
      `;
      this.probBarsContainer.appendChild(row);
    }
  }

  initEvents() {
    this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
    this.canvas.addEventListener('mousemove', (e) => this.draw(e));
    window.addEventListener('mouseup', () => this.stopDrawing());
    this.canvas.addEventListener('mouseleave', () => this.stopDrawing());

    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.startDrawing(e.touches[0]);
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      this.draw(e.touches[0]);
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.stopDrawing();
    }, { passive: false });

    this.clearBtn.addEventListener('click', () => this.clear());
    this.predictBtn.addEventListener('click', () => this.predict());

    const pills = document.querySelectorAll('.pill-btn');
    pills.forEach((pill) => {
      pill.addEventListener('click', () => {
        pills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        this.currentModel = pill.dataset.model;
        this.updateModelUI();
        if (this.hasDrawn) this.predict();
      });
    });
  }

  updateModelUI() {
    document.body.classList.remove('model-cnn', 'model-mlp', 'model-sklearn', 'model-compare');
    document.body.classList.add(`model-${this.currentModel}`);

    if (this.currentModel === 'compare') {
      this.singlePredCard.classList.add('hidden');
      this.comparisonCards.classList.remove('hidden');
      this.probTitle.textContent = 'Digit Probabilities (CNN)';
      this.footerDesc.innerHTML = 'Comparing <strong>PyTorch CNN (99.11%)</strong>, <strong>PyTorch MLP (97.68%)</strong>, and <strong>Scikit-Learn Logistic Regression (92.59%)</strong> side-by-side.';
    } else {
      this.singlePredCard.classList.remove('hidden');
      this.comparisonCards.classList.add('hidden');

      if (this.currentModel === 'cnn') {
        this.activeModelTag.textContent = 'PyTorch CNN (99.11%)';
        this.activeModelTag.className = 'model-tag';
        this.probTitle.textContent = 'Digit Probabilities (CNN)';
        this.footerDesc.innerHTML = 'Currently running: <strong>PyTorch CNN</strong> (99.11% accuracy, ~97k parameters). Deep 2D spatial convolutions.';
      } else if (this.currentModel === 'mlp') {
        this.activeModelTag.textContent = 'PyTorch MLP (97.68%)';
        this.activeModelTag.className = 'model-tag tag-mlp';
        this.probTitle.textContent = 'Digit Probabilities (MLP)';
        this.footerDesc.innerHTML = 'Currently running: <strong>PyTorch MLP</strong> (97.68% accuracy, ~218k parameters). Multi-layer dense feed-forward network.';
      } else {
        this.activeModelTag.textContent = 'Scikit-Learn ML (92.59%)';
        this.activeModelTag.className = 'model-tag tag-sklearn';
        this.probTitle.textContent = 'Digit Probabilities (Logistic Regression)';
        this.footerDesc.innerHTML = 'Currently running: <strong>Scikit-Learn Logistic Regression</strong> (92.59% accuracy, 7.8k parameters). Classical linear softmax model.';
      }
    }
  }

  getCanvasCoords(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
      y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
    };
  }

  startDrawing(e) {
    this.isDrawing = true;
    this.hasDrawn = true;
    this.canvasHint.classList.add('hidden');

    const coords = this.getCanvasCoords(e);
    this.lastX = coords.x;
    this.lastY = coords.y;

    this.ctx.beginPath();
    this.ctx.arc(this.lastX, this.lastY, this.ctx.lineWidth / 2, 0, Math.PI * 2);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fill();
  }

  draw(e) {
    if (!this.isDrawing) return;
    const coords = this.getCanvasCoords(e);
    this.ctx.beginPath();
    this.ctx.moveTo(this.lastX, this.lastY);
    this.ctx.lineTo(coords.x, coords.y);
    this.ctx.stroke();
    this.lastX = coords.x;
    this.lastY = coords.y;
  }

  stopDrawing() {
    this.isDrawing = false;
  }

  clear() {
    this.initCanvas();
    this.hasDrawn = false;
    this.canvasHint.classList.remove('hidden');

    this.predictionDisplay.textContent = '?';
    this.predictionDisplay.className = 'prediction-display';
    this.confidenceDisplay.textContent = '—';

    this.compareCnnDigit.textContent = '?';
    this.compareCnnConf.textContent = 'Conf: —';
    this.compareMlpDigit.textContent = '?';
    this.compareMlpConf.textContent = 'Conf: —';
    this.compareSklearnDigit.textContent = '?';
    this.compareSklearnConf.textContent = 'Conf: —';

    for (let i = 0; i < 10; i++) {
      const row = document.getElementById(`prob-row-${i}`);
      const fill = document.getElementById(`prob-fill-${i}`);
      const val = document.getElementById(`prob-val-${i}`);
      if (row) row.classList.remove('active');
      if (fill) fill.style.width = '0%';
      if (val) val.textContent = '0.0%';
    }
  }

  getBestPrediction(probabilities) {
    let bestDigit = 0;
    let bestProb = probabilities[0];
    for (let i = 1; i < 10; i++) {
      if (probabilities[i] > bestProb) {
        bestProb = probabilities[i];
        bestDigit = i;
      }
    }
    return { digit: bestDigit, prob: bestProb };
  }

  predict() {
    if (!this.hasDrawn) {
      this.predictionDisplay.textContent = '?';
      this.confidenceDisplay.textContent = 'Draw first!';
      return;
    }

    const input = MNISTPreprocessor.process(this.canvas, this.previewCanvas);
    if (!input) {
      this.predictionDisplay.textContent = '?';
      this.confidenceDisplay.textContent = 'Draw clearer digit';
      return;
    }

    // Run inference across all 3 engines
    const cnnProbs = this.cnn.predict(input);
    const mlpProbs = this.mlp.predict(input);
    const sklearnProbs = this.sklearn.predict(input);

    const cnnResult = this.getBestPrediction(cnnProbs);
    const mlpResult = this.getBestPrediction(mlpProbs);
    const sklearnResult = this.getBestPrediction(sklearnProbs);

    // Update 3-Way Comparison View
    this.compareCnnDigit.textContent = cnnResult.digit;
    this.compareCnnConf.textContent = `Conf: ${(cnnResult.prob * 100).toFixed(1)}%`;
    this.compareMlpDigit.textContent = mlpResult.digit;
    this.compareMlpConf.textContent = `Conf: ${(mlpResult.prob * 100).toFixed(1)}%`;
    this.compareSklearnDigit.textContent = sklearnResult.digit;
    this.compareSklearnConf.textContent = `Conf: ${(sklearnResult.prob * 100).toFixed(1)}%`;

    let activeProbs = cnnProbs;
    let activeResult = cnnResult;

    if (this.currentModel === 'mlp') {
      activeProbs = mlpProbs;
      activeResult = mlpResult;
      this.predictionDisplay.textContent = mlpResult.digit;
      this.confidenceDisplay.textContent = `${(mlpResult.prob * 100).toFixed(1)}%`;
      this.predictionDisplay.className = 'prediction-display predicted-mlp';
    } else if (this.currentModel === 'sklearn') {
      activeProbs = sklearnProbs;
      activeResult = sklearnResult;
      this.predictionDisplay.textContent = sklearnResult.digit;
      this.confidenceDisplay.textContent = `${(sklearnResult.prob * 100).toFixed(1)}%`;
      this.predictionDisplay.className = 'prediction-display predicted-sklearn';
    } else {
      this.predictionDisplay.textContent = cnnResult.digit;
      this.confidenceDisplay.textContent = `${(cnnResult.prob * 100).toFixed(1)}%`;
      this.predictionDisplay.className = 'prediction-display predicted-cnn';
    }

    // Update Probability Bars
    for (let i = 0; i < 10; i++) {
      const row = document.getElementById(`prob-row-${i}`);
      const fill = document.getElementById(`prob-fill-${i}`);
      const val = document.getElementById(`prob-val-${i}`);
      const p = activeProbs[i];

      if (fill) fill.style.width = `${(p * 100).toFixed(1)}%`;
      if (val) val.textContent = `${(p * 100).toFixed(1)}%`;

      if (row) {
        if (i === activeResult.digit) {
          row.classList.add('active');
        } else {
          row.classList.remove('active');
        }
      }
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.digitApp = new DigitApp();
});
