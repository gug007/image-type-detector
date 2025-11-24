/* eslint-disable @typescript-eslint/no-require-imports */
const tf = require('@tensorflow/tfjs');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '../src/app/images');
const OUTPUT_DIR = path.join(__dirname, '../public/model');
const DOC_DIR = path.join(DATA_DIR, 'document');
const REGULAR_DIR = path.join(DATA_DIR, 'regular_iamge');

const IMAGE_SIZE = 224;
const NUM_CLASSES = 2; // 0: document, 1: regular

// Custom IO Handler for Node.js filesystem saving
class NodeFileSystemSaveHandler {
  constructor(dir) {
    this.dir = dir;
  }

  async save(modelArtifacts) {
    if (!fs.existsSync(this.dir)) {
      fs.mkdirSync(this.dir, { recursive: true });
    }

    // Handle weightsManifest construction if missing
    let weightsManifest = modelArtifacts.weightsManifest;
    if (!weightsManifest && modelArtifacts.weightSpecs) {
       weightsManifest = [{
         paths: ['./weights.bin'],
         weights: modelArtifacts.weightSpecs
       }];
    }
    
    const modelJson = {
      modelTopology: modelArtifacts.modelTopology,
      format: modelArtifacts.format,
      generatedBy: modelArtifacts.generatedBy,
      convertedBy: modelArtifacts.convertedBy,
      weightsManifest: weightsManifest,
    };
    
    if (modelArtifacts.weightData != null) {
      // Save weights binary
      const weightsPath = path.join(this.dir, 'weights.bin');
      fs.writeFileSync(weightsPath, Buffer.from(modelArtifacts.weightData));
      
      // Update manifest paths to be relative to model.json
      if (modelJson.weightsManifest) {
        modelJson.weightsManifest.forEach(manifest => {
          manifest.paths = ['./weights.bin'];
        });
      }
    }

    fs.writeFileSync(
      path.join(this.dir, 'model.json'),
      JSON.stringify(modelJson, null, 2)
    );
    
    // Also save metadata.json for class names if we want
    const metadata = {
      labels: ['document', 'regular']
    };
    fs.writeFileSync(
      path.join(this.dir, 'metadata.json'),
      JSON.stringify(metadata, null, 2)
    );

    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: JSON.stringify(modelArtifacts.modelTopology).length,
        weightSpecsBytes: 0,
        weightDataBytes: modelArtifacts.weightData ? modelArtifacts.weightData.byteLength : 0,
      }
    };
  }
}

async function loadImages() {
  const images = [];
  const labels = [];

  const docFiles = fs.readdirSync(DOC_DIR).filter(f => f.endsWith('.jpg'));
  const regFiles = fs.readdirSync(REGULAR_DIR).filter(f => f.endsWith('.jpg'));

  console.log(`Found ${docFiles.length} document images`);
  console.log(`Found ${regFiles.length} regular images`);

  for (const file of docFiles) {
    const imgPath = path.join(DOC_DIR, file);
    const img = await loadImage(imgPath);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    const tensor = tf.browser.fromPixels(canvas)
      .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
    images.push(tensor);
    labels.push(0); // 0 for document
  }

  for (const file of regFiles) {
    const imgPath = path.join(REGULAR_DIR, file);
    const img = await loadImage(imgPath);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    const tensor = tf.browser.fromPixels(canvas)
      .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
    images.push(tensor);
    labels.push(1); // 1 for regular
  }

  return {
    images: tf.concat(images),
    labels: tf.oneHot(tf.tensor1d(labels, 'int32'), NUM_CLASSES)
  };
}

async function train() {
  // Ensure CPU backend is used if others fail, but usually auto-detected.
  await tf.setBackend('cpu');
  console.log('Using backend:', tf.getBackend());

  const data = await loadImages();
  console.log('Data loaded');
  
  // Load MobileNet
  console.log('Loading MobileNet...');
  // Need to use global fetch or polyfill if not available, but Node 18+ has fetch.
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  
  // Create a new model
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  const truncatedMobileNet = tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  
  const model = tf.sequential();
  model.add(truncatedMobileNet);
  model.add(tf.layers.globalAveragePooling2d({}));
  model.add(tf.layers.dense({units: 100, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.5})); // Add dropout to reduce overfitting
  model.add(tf.layers.dense({units: NUM_CLASSES, activation: 'softmax'}));

  // Freeze the MobileNet layers
  for (const layer of truncatedMobileNet.layers) {
    layer.trainable = false;
  }

  model.compile({
    optimizer: tf.train.adam(0.00005), // Lower learning rate for fine-tuning
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  console.log('Training model...');
  await model.fit(data.images, data.labels, {
    epochs: 100, // More epochs
    batchSize: 4,
    validationSplit: 0.0,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc.toFixed(4)}`)
    }
  });

  console.log('Saving model...');
  // Use custom IO handler
  await model.save(new NodeFileSystemSaveHandler(OUTPUT_DIR));
  console.log('Model saved to ' + OUTPUT_DIR);
}

train().catch(console.error);
