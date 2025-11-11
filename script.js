let model;
const upload = document.getElementById('upload');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

async function loadModel() {
  result.textContent = 'Loading MobileNetV2 model...';
  
  try {
    // Option 1: Use TensorFlow Hub MobileNetV2 (recommended)
    model = await tf.loadGraphModel(
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1', 
      { fromTFHub: true }
    );
    
    // If the above fails, you can also try MobileNetV1 which is more reliably available:
    // model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    
    result.textContent = 'Model loaded. Upload an image to classify.';
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    result.textContent = 'Error loading model. Trying alternative...';
    
    // Fallback to MobileNetV1
    try {
      model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
      result.textContent = 'MobileNetV1 loaded. Upload an image to classify.';
    } catch (fallbackError) {
      console.error('Fallback also failed:', fallbackError);
      result.textContent = 'Failed to load any model. Check console for details.';
    }
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
        // Preprocess the image
        const tensor = tf.browser.fromPixels(preview)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .expandDims(0);
        
        let predictions;
        
        // Check if it's a GraphModel (MobileNetV2) or LayersModel (MobileNetV1)
        if (model instanceof tf.GraphModel) {
          // For GraphModel (MobileNetV2) - normalize to [-1, 1]
          const normalized = tensor.div(127.5).sub(1);
          predictions = model.predict(normalized);
          normalized.dispose();
        } else {
          // For LayersModel (MobileNetV1) - normalize to [0, 1] and use imagenet preprocessing
          const normalized = tensor.div(255);
          const preprocessed = tf.sub(normalized, tf.tensor1d([0.485, 0.456, 0.406]))
            .div(tf.tensor1d([0.229, 0.224, 0.225]));
          predictions = model.predict(preprocessed);
          preprocessed.dispose();
        }

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

        // Clean up tensors
        tensor.dispose();
        predictions.dispose();

      } catch (error) {
        console.error('Error during prediction:', error);
        result.textContent = 'Error analyzing image. Check console for details.';
      }
    };
  };
  reader.readAsDataURL(file);
});

// Load model when page loads
loadModel();
