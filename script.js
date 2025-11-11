let model;
const upload = document.getElementById('upload');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

async function loadModel() {
  result.textContent = 'Loading MobileNetV2 model...';
  
  try {
    // Use loadGraphModel for MobileNetV2 instead of loadLayersModel
    // model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1', { fromTFHub: true });
    model = await tf.loadGraphModel('https://storage.googleapis.com/tfhub-tfjs-modules/google/tf2-preview/mobilenet_v2/classification/4/default/1', { fromTFHub: true });
    result.textContent = 'Model loaded. Upload an image to classify.';
  } catch (error) {
    console.error('Error loading model:', error);
    result.textContent = 'Error loading model. Check console for details.';
  }
}

upload.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = async function () {
    preview.src = reader.result;
    result.textContent = 'Analyzing...';

    preview.onload = async () => {
      if (!model) {
        result.textContent = 'Model not loaded yet. Please wait.';
        return;
      }

      try {
        // Preprocess the image for MobileNetV2
        const tensor = tf.browser.fromPixels(preview)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .expandDims(0);
        
        // Normalize to [-1, 1] (MobileNetV2 expects this range)
        const normalized = tensor.div(127.5).sub(1);

        // Get predictions
        const predictions = model.predict(normalized);
        const data = await predictions.data();

        // Get top 5 predictions
        const top5 = Array.from(data)
          .map((probability, index) => ({ probability, className: index }))
          .sort((a, b) => b.probability - a.probability)
          .slice(0, 5);

        // Display results
        result.innerHTML = top5.map((p, i) => `
          <div style="margin: 10px 0;">
            #${i + 1}: Class ${p.className} (${(p.probability * 100).toFixed(2)}%)
            <div class="bar" style="width:${Math.min(p.probability * 100, 100)}%;"></div>
          </div>`).join('');

        // Clean up tensors
        tensor.dispose();
        normalized.dispose();
        predictions.dispose();

      } catch (error) {
        console.error('Error during prediction:', error);
        result.textContent = 'Error analyzing image. Check console for details.';
      }
    };
  };
  reader.readAsDataURL(file);
});

loadModel();
