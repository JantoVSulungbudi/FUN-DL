async function loadClassLabels() {
  try {
    // Try primary source first
    const response = await fetch('https://storage.googleapis.com/tfjs-models/assets/imagenet/labels.json');
    classLabels = await response.json();
  } catch (error) {
    console.log('Primary labels failed, trying alternative...');
    try {
      // Alternative source from GitHub
      const response = await fetch('https://raw.githubusercontent.com/tensorflow/tfjs-models/master/mobilenet/src/imagenet_classes.ts');
      const text = await response.text();
      
      // Parse the TypeScript file to extract labels
      const lines = text.split('\n');
      const labels = [];
      let inLabelsArray = false;
      
      for (const line of lines) {
        if (line.includes('export const IMAGENET_CLASSES = {')) {
          inLabelsArray = true;
          continue;
        }
        if (inLabelsArray && line.includes('};')) {
          break;
        }
        if (inLabelsArray && line.includes(':')) {
          const match = line.match(/\d+:\s*'([^']+)'/);
          if (match) {
            labels.push(match[1]);
          }
        }
      }
      
      classLabels = labels;
    } catch (fallbackError) {
      console.error('All label sources failed:', fallbackError);
      // Final fallback: create generic labels
      classLabels = Array.from({ length: 1000 }, (_, i) => `Class ${i}`);
    }
  }
}
