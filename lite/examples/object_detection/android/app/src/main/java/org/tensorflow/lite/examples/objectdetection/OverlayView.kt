/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: ObjectDetectorResult? = null
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var scaleFactor: Float = 1f
    private var outputWidth = 0
    private var outputHeight = 0
    private var outputRotation = 0

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        results = null
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        val detectionResults = results?.detections() ?: return

        detectionResults.map { detection ->
            val boxRect = RectF(
                detection.boundingBox().left,
                detection.boundingBox().top,
                detection.boundingBox().right,
                detection.boundingBox().bottom
            )

            val matrix = Matrix()
            matrix.postTranslate(-outputWidth / 2f, -outputHeight / 2f)
            matrix.postRotate(outputRotation.toFloat())
            if (outputRotation == 90 || outputRotation == 270) {
                matrix.postTranslate(outputHeight / 2f, outputWidth / 2f)
            } else {
                matrix.postTranslate(outputWidth / 2f, outputHeight / 2f)
            }
            matrix.mapRect(boxRect)
            boxRect
        }.forEachIndexed { index, boxRect ->
            val top = boxRect.top * scaleFactor
            val bottom = boxRect.bottom * scaleFactor
            val left = boxRect.left * scaleFactor
            val right = boxRect.right * scaleFactor

            canvas.drawRect(RectF(left, top, right, bottom), boxPaint)

            val category = detectionResults[index].categories().firstOrNull() ?: return@forEachIndexed
            val categoryName = category.categoryName()
                ?.takeUnless { it.isBlank() }
                ?: category.displayName()
                ?.takeUnless { it.isBlank() }
                ?: "Object"
            val drawableText = "$categoryName ${String.format("%.2f", category.score())}"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
        }
    }

    fun setResults(
      detectionResults: ObjectDetectorResult,
      imageHeight: Int,
      imageWidth: Int,
      imageRotation: Int
    ) {
        results = detectionResults
        outputWidth = imageWidth
        outputHeight = imageHeight
        outputRotation = imageRotation

        val rotatedWidthHeight = when (imageRotation) {
            0, 180 -> Pair(imageWidth, imageHeight)
            90, 270 -> Pair(imageHeight, imageWidth)
            else -> Pair(imageWidth, imageHeight)
        }

        scaleFactor = max(
            width * 1f / rotatedWidthHeight.first,
            height * 1f / rotatedWidthHeight.second
        )
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
