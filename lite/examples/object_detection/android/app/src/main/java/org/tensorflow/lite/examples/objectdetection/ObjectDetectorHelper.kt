/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult

class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector?.close()
        objectDetector = null
    }

    fun setupObjectDetector() {
        val modelName = when (currentModel) {
            MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
            MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
            MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
            MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
            else -> "mobilenetv1.tflite"
        }

        val delegate = when (currentDelegate) {
            DELEGATE_GPU -> Delegate.GPU
            DELEGATE_NNAPI -> {
                objectDetectorListener?.onError(
                    "NNAPI is not supported in this MediaPipe build. Falling back to CPU."
                )
                Delegate.CPU
            }
            else -> Delegate.CPU
        }

        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(modelName)
            .setDelegate(delegate)
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)
            .build()

        try {
            clearObjectDetector()
            objectDetector = ObjectDetector.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details."
            )
            Log.e(TAG, "MediaPipe failed to load model with error: ${e.message}", e)
        } catch (e: RuntimeException) {
            objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details."
            )
            Log.e(TAG, "MediaPipe runtime failed to initialize detector: ${e.message}", e)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        val detector = objectDetector ?: return
        val startTime = SystemClock.uptimeMillis()

        try {
            val mpImage = BitmapImageBuilder(image).build()
            val imageProcessingOptions = ImageProcessingOptions.builder()
                .setRotationDegrees(imageRotation)
                .build()

            val results = detector.detect(mpImage, imageProcessingOptions)
            val inferenceTime = SystemClock.uptimeMillis() - startTime
            objectDetectorListener?.onResults(
                results,
                inferenceTime,
                image.height,
                image.width,
                imageRotation
            )
        } catch (e: RuntimeException) {
            objectDetectorListener?.onError(
                "Object detection failed. See error logs for details."
            )
            Log.e(TAG, "MediaPipe object detection failed: ${e.message}", e)
        }
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
          results: ObjectDetectorResult,
          inferenceTime: Long,
          imageHeight: Int,
          imageWidth: Int,
          imageRotation: Int
        )
    }

    companion object {
        private const val TAG = "ObjectDetectorHelper"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
    }
}
