# MobileNetV2 Image Classifier (TensorFlow.js)

This is a simple **client-side** demo that runs MobileNetV2 (pre-trained on ImageNet)
entirely in the browser using **TensorFlow.js**.

## 🚀 How to Use

1. Upload the three files to a GitHub repository, for example `mobilenetv2-demo`.
2. Go to **Settings → Pages → Build and deployment → Source → Deploy from branch**.
3. Choose `main` branch and root (`/` folder).
4. Wait for a few minutes until the site is live at:

   ```
   https://your-username.github.io/mobilenetv2-demo/
   ```

5. Open the site, upload an image, and see predictions appear below it.

## 🧠 Model
The demo uses **MobileNetV2 (140×224)** pretrained on ImageNet from TensorFlow Hub:

```
https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_140_224/classification/5
```

You can later replace this with your own model by modifying `MODEL_URL` in `script.js`.
