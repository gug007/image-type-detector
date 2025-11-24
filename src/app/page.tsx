"use client";

import { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export default function Home() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);

  useEffect(() => {
    async function loadModel() {
      try {
        await tf.ready();
        const loadedModel = await tf.loadLayersModel("/model/model.json");
        setModel(loadedModel);
        setLoading(false);
        console.log("Model loaded");
      } catch (err) {
        console.error("Failed to load model", err);
        setLoading(false);
      }
    }
    loadModel();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && model) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.onload = async () => {
        tf.tidy(() => {
          const tensor = tf.browser
            .fromPixels(img)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();

          const predictions = model.predict(tensor) as tf.Tensor;
          const data = predictions.dataSync();
          // 0: document, 1: regular
          const docScore = data[0];
          const regScore = data[1];

          if (docScore > regScore) {
            setPrediction("Document");
            setConfidence(docScore);
          } else {
            setPrediction("Regular Image");
            setConfidence(regScore);
          }
        });
      };
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 p-4">
      <h1 className="text-2xl font-bold">Image Type Detector</h1>

      {loading && <p>Loading model...</p>}

      {!loading && !model && (
        <p className="text-red-500">Failed to load model</p>
      )}

      {!loading && model && (
        <div className="flex flex-col items-center gap-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-slate-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-violet-50 file:text-violet-700
              hover:file:bg-violet-100
            "
          />

          {prediction && (
            <div className="text-center p-4 border rounded-lg bg-gray-50 dark:bg-gray-800">
              <p className="text-xl font-semibold text-white">
                Detected: {prediction}
              </p>
              {confidence !== null && (
                <p className="text-sm text-gray-500">
                  Confidence: {(confidence * 100).toFixed(1)}%
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
