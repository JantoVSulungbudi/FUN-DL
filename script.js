let model;
const upload = document.getElementById('upload');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

async function loadModel() {
  result.textContent = 'Loading MobileNet model...';
  
  try {
    // Use MobileNetV1 which has a reliable URL
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    result.textContent = 'Model loaded. Upload an image to classify.';
  } catch (error) {
    console.error('Error loading model:', error);
    result.textContent = 'Error loading model. Please refresh the page.';
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
        // Preprocess image for MobileNetV1
        const tensor = tf.browser.fromPixels(preview)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .expandDims(0)
          .div(255); // Normalize to [0, 1]

        // Get predictions
        const predictions = model.predict(tensor);
        const data = await predictions.data();

        // Get top 5 predictions
        const top5 = Array.from(data)
          .map((probability, index) => ({ probability, className: index }))
          .sort((a, b) => b.probability - a.probability)
          .slice(0, 5);

        // Display results
        result.innerHTML = '<strong>Top 5 Predictions:</strong><br>' + 
          top5.map((p, i) => `
            <div style="margin: 10px 0;">
              #${i + 1}: Class ${p.className} (${(p.probability * 100).toFixed(2)}%)
              <div class="bar" style="width:${Math.min(p.probability * 100, 100)}%;"></div>
            </div>`).join('');

        // Clean up
        tensor.dispose();
        predictions.dispose();

      } catch (error) {
        console.error('Error during prediction:', error);
        result.textContent = 'Error analyzing image.';
      }
    };
  };
  reader.readAsDataURL(file);
});

loadModel();
