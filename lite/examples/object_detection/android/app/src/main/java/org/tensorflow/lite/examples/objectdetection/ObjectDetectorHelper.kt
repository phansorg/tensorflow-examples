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
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.metadata.MetadataExtractor

class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        interpreter?.close()
        interpreter = null
    }

    fun setupObjectDetector() {
        val modelName = when (currentModel) {
            MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
            MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
            MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
            MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
            else -> "mobilenetv1.tflite"
        }

        try {
            clearObjectDetector()

            val litertBuffer = FileUtil.loadMappedFile(context, modelName)
            labels = getModelMetadata(litertBuffer)
            interpreter = Interpreter(litertBuffer, Interpreter.Options().apply {
                numThreads = this@ObjectDetectorHelper.numThreads
                useNNAPI = currentDelegate == DELEGATE_NNAPI
            })
        } catch (e: Exception) {
            objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details."
            )
            Log.e(TAG, "LiteRT failed to initialize detector: ${e.message}", e)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        val currentInterpreter = interpreter ?: run {
            setupObjectDetector()
            interpreter ?: return
        }

        val startTime = SystemClock.uptimeMillis()

        try {
            val (_, inputHeight, inputWidth, _) = currentInterpreter.getInputTensor(0).shape()
            val tensorImage = createTensorImage(
                bitmap = image,
                width = inputWidth,
                height = inputHeight,
                rotationDegrees = imageRotation
            )

            val outputs = runDetection(tensorImage)
            val detections = getDetections(
                locations = outputs.locations,
                categories = outputs.categories,
                scores = outputs.scores,
                imageWidth = inputWidth,
                scaleRatio = inputHeight.toFloat() / tensorImage.height
            )

            val inferenceTime = SystemClock.uptimeMillis() - startTime
            objectDetectorListener?.onResults(
                detections,
                inferenceTime,
                tensorImage.height,
                tensorImage.width
            )
        } catch (e: Exception) {
            objectDetectorListener?.onError(
                "Object detection failed. See error logs for details."
            )
            Log.e(TAG, "LiteRT inference failed: ${e.message}", e)
        }
    }

    private fun runDetection(tensorImage: TensorImage): DetectionOutputs {
        val currentInterpreter = interpreter ?: error("Interpreter is not initialized.")
        val locationOutputShape = currentInterpreter.getOutputTensor(0).shape()
        val categoryOutputShape = currentInterpreter.getOutputTensor(1).shape()
        val scoreOutputShape = currentInterpreter.getOutputTensor(2).shape()

        val locationOutputBuffer =
            FloatBuffer.allocate(locationOutputShape[1] * locationOutputShape[2])
        val categoryOutputBuffer = FloatBuffer.allocate(categoryOutputShape[1])
        val scoreOutputBuffer = FloatBuffer.allocate(scoreOutputShape[1])

        currentInterpreter.runForMultipleInputsOutputs(
            arrayOf(tensorImage.tensorBuffer.buffer),
            mapOf(
                0 to locationOutputBuffer,
                1 to categoryOutputBuffer,
                2 to scoreOutputBuffer,
            )
        )

        locationOutputBuffer.rewind()
        categoryOutputBuffer.rewind()
        scoreOutputBuffer.rewind()

        return DetectionOutputs(
            locations = FloatArray(locationOutputBuffer.capacity()).also { locationOutputBuffer.get(it) },
            categories = FloatArray(categoryOutputBuffer.capacity()).also { categoryOutputBuffer.get(it) },
            scores = FloatArray(scoreOutputBuffer.capacity()).also { scoreOutputBuffer.get(it) },
        )
    }

    private fun createTensorImage(
      bitmap: Bitmap,
      width: Int,
      height: Int,
      rotationDegrees: Int
    ): TensorImage {
        val rotation = -rotationDegrees / 90
        val scaledBitmap = fitCenterBitmap(bitmap, width, height)
        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(rotation)).build()
        return imageProcessor.process(TensorImage.fromBitmap(scaledBitmap))
    }

    private fun fitCenterBitmap(
      originalBitmap: Bitmap,
      width: Int,
      height: Int
    ): Bitmap {
        val bitmapWithBackground = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmapWithBackground)
        canvas.drawColor(Color.TRANSPARENT)

        val scale = height.toFloat() / originalBitmap.height
        val scaledWidth = width * scale
        val scaledBitmap = Bitmap.createScaledBitmap(
            originalBitmap.copy(Bitmap.Config.ARGB_8888, true),
            scaledWidth.toInt(),
            height,
            true
        )

        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        val left = (width - scaledWidth) / 2
        canvas.drawBitmap(scaledBitmap, left, 0f, paint)
        return bitmapWithBackground
    }

    private fun getModelMetadata(litertBuffer: ByteBuffer): List<String> {
        val metadataExtractor = MetadataExtractor(litertBuffer)
        if (!metadataExtractor.hasMetadata()) {
            return emptyList()
        }
        val inputStream = metadataExtractor.getAssociatedFile("labelmap.txt")
        return readFileInputStream(inputStream)
    }

    private fun readFileInputStream(inputStream: InputStream): List<String> {
        val reader = BufferedReader(InputStreamReader(inputStream))
        val lines = mutableListOf<String>()
        var line: String?
        while (reader.readLine().also { line = it } != null) {
            lines.add(line.orEmpty())
        }
        reader.close()
        return lines
    }

    private fun getDetections(
      locations: FloatArray,
      categories: FloatArray,
      scores: FloatArray,
      imageWidth: Int,
      scaleRatio: Float
    ): List<Detection> {
        val boundingBoxList = getBoundingBoxList(locations, imageWidth, scaleRatio)
        val resultCount = minOf(maxResults, scores.size, categories.size, boundingBoxList.size)

        return buildList {
            for (i in 0 until resultCount) {
                val categoryIndex = categories[i].toInt()
                val label = labels.getOrNull(categoryIndex) ?: categoryIndex.toString()
                add(
                    Detection(
                        label = label,
                        boundingBox = boundingBoxList[i],
                        score = scores[i]
                    )
                )
            }
        }
            .filter { !it.boundingBox.isEmpty && it.score >= threshold }
            .sortedByDescending { it.score }
    }

    private fun getBoundingBoxList(
      locations: FloatArray,
      imageWidth: Int,
      scaleRatio: Float
    ): Array<RectF> {
        val boundingBoxList = Array(locations.size / 4) { RectF() }
        val actualWidth = imageWidth * scaleRatio
        val padding = (imageWidth - imageWidth * scaleRatio) / 2

        for (i in boundingBoxList.indices) {
            val topRatio = locations[i * 4]
            val leftRatio = locations[i * 4 + 1]
            val bottomRatio = locations[i * 4 + 2]
            val rightRatio = locations[i * 4 + 3]

            val top = topRatio.coerceIn(0f, 1f)
            val left = ((leftRatio * imageWidth - padding) / actualWidth).coerceIn(0f, 1f)
            val bottom = bottomRatio.coerceIn(top, 1f)
            val right = ((rightRatio * imageWidth - padding) / actualWidth).coerceIn(left, 1f)

            boundingBoxList[i] = RectF(left, top, right, bottom)
        }

        return boundingBoxList
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
          results: List<Detection>,
          inferenceTime: Long,
          imageHeight: Int,
          imageWidth: Int
        )
    }

    data class Detection(
      val label: String,
      val boundingBox: RectF,
      val score: Float
    )

    private data class DetectionOutputs(
      val locations: FloatArray,
      val categories: FloatArray,
      val scores: FloatArray
    )

    companion object {
        private const val TAG = "ObjectDetectorHelper"

        const val DELEGATE_CPU = 0
        const val DELEGATE_NNAPI = 1
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
    }
}
