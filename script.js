let model;
const upload = document.getElementById('upload');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

async function loadModel() {
  result.textContent = 'Loading MobileNetV2 model...';
  model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json');
  result.textContent = 'Model loaded. Upload an image to classify.';
}

upload.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  const reader = new FileReader();

  reader.onload = async function () {
    preview.src = reader.result;
    result.textContent = 'Analyzing...';

    preview.onload = async () => {
      const tensor = tf.browser.fromPixels(preview)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255))
        .expandDims();

      const predictions = model.predict(tensor);
      const data = await predictions.data();

      const top5 = Array.from(data)
        .map((p, i) => ({ probability: p, className: i }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 5);

      result.innerHTML = top5
        .map((p, i) => `#${i + 1}: Class ${p.className} (${(p.probability * 100).toFixed(2)}%)`)
        .join('<br>');
    };
  };
  reader.readAsDataURL(file);
});

loadModel();
