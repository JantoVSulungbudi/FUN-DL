const upload = document.getElementById('upload');
const preview = document.getElementById('preview');
const statusEl = document.getElementById('status');
const output = document.getElementById('output');

let model;
let labels;

// Pretrained MobileNetV2 model from TFHub
const MODEL_URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_140_224/classification/5';
const LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json';

// Load ImageNet labels
async function loadLabels() {
  const res = await fetch(LABELS_URL);
  const json = await res.json();
  const arr = new Array(1000);
  for (let i in json) arr[i] = json[i][1];
  return arr;
}

// Preprocess the image to 224x224 and normalize [-1,1]
function preprocess(img) {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(img).resizeBilinear([224, 224]).toFloat();
    const normalized = tensor.div(127.5).sub(1);
    return normalized.expandDims(0);
  });
}

async function init() {
  statusEl.textContent = 'Loading model...';
  model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
  labels = await loadLabels();
  statusEl.textContent = 'Model ready! Upload an image.';
}

upload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);

  preview.onload = async () => {
    statusEl.textContent = 'Classifying...';
    const input = preprocess(preview);
    const logits = model.predict(input);
    const probs = await tf.softmax(logits).data();
    input.dispose();
    logits.dispose();

    // Get top 5 predictions
    const top = Array.from(probs)
      .map((p, i) => ({ i, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 5);

    let html = `<table><tr><th>Rank</th><th>Label</th><th>Probability</th></tr>`;
    for (let k = 0; k < top.length; k++) {
      html += `<tr><td>${k + 1}</td><td>${labels[top[k].i]}</td><td>${(top[k].p * 100).toFixed(2)}%</td></tr>`;
    }
    html += `</table>`;
    output.innerHTML = html;
    statusEl.textContent = 'Done.';
  };
});

init();
