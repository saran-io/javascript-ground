async function predictImage() {
    // Load the TensorFlow.js library
    const tf = require('@tensorflow/tfjs-node');
    const fs = require('fs');
  
    // Load the image
    const image = tf.node.decodeImage(fs.readFileSync('image.jpg'));
  
    // Resize the image to 224x224 pixels
    const resizedImage = tf.image.resizeBilinear(image, [224, 224]);
  
    // Normalize the pixel values to be between -1 and 1
    const normalizedImage = resizedImage.div(255 / 2).sub(1);
  
    // Load the pre-trained MobileNet model
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  
    // Make a prediction on the image
    const prediction = model.predict(normalizedImage.reshape([1, 224, 224, 3]));
  
    // Get the top 5 predicted classes and their probabilities
    const topK = tf.topk(prediction, 5);
    const values = await topK.values.data();
    const indices = await topK.indices.data();
  
    // Print the top 5 predicted classes and their probabilities
    console.log('Top 5 predictions:');
    for (let i = 0; i < 5; i++) {
      console.log(`Class ${indices[i]}: ${(values[i] * 100).toFixed(2)}%`);
    }
  }
  