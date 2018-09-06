package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_image_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.INT32
import wumo.sim.tensorflow.types.UINT8

object image_ops {
  interface API {
    fun adjustContrast(images: Output, contrastFactor: Output, minValue: Output, maxValue: Output, name: String = "AdjustContrast"): Output {
      return gen_image_ops.adjustContrast(images, contrastFactor, minValue, maxValue, name)
    }
    
    fun adjustContrastv2(images: Output, contrastFactor: Output, name: String = "AdjustContrastv2"): Output {
      return gen_image_ops.adjustContrastv2(images, contrastFactor, name)
    }
    
    fun adjustHue(images: Output, delta: Output, name: String = "AdjustHue"): Output {
      return gen_image_ops.adjustHue(images, delta, name)
    }
    
    fun adjustSaturation(images: Output, scale: Output, name: String = "AdjustSaturation"): Output {
      return gen_image_ops.adjustSaturation(images, scale, name)
    }
    
    fun cropAndResize(image: Output, boxes: Output, boxInd: Output, cropSize: Output, method: String = "bilinear", extrapolationValue: Float = 0.0f, name: String = "CropAndResize"): Output {
      return gen_image_ops.cropAndResize(image, boxes, boxInd, cropSize, method, extrapolationValue, name)
    }
    
    fun cropAndResizeGradBoxes(grads: Output, image: Output, boxes: Output, boxInd: Output, method: String = "bilinear", name: String = "CropAndResizeGradBoxes"): Output {
      return gen_image_ops.cropAndResizeGradBoxes(grads, image, boxes, boxInd, method, name)
    }
    
    fun cropAndResizeGradImage(grads: Output, boxes: Output, boxInd: Output, imageSize: Output, t: DataType<*>, method: String = "bilinear", name: String = "CropAndResizeGradImage"): Output {
      return gen_image_ops.cropAndResizeGradImage(grads, boxes, boxInd, imageSize, t, method, name)
    }
    
    fun decodeAndCropJpeg(contents: Output, cropWindow: Output, channels: Long = 0L, ratio: Long = 1L, fancyUpscaling: Boolean = true, tryRecoverTruncated: Boolean = false, acceptableFraction: Float = 1.0f, dctMethod: String = "", name: String = "DecodeAndCropJpeg"): Output {
      return gen_image_ops.decodeAndCropJpeg(contents, cropWindow, channels, ratio, fancyUpscaling, tryRecoverTruncated, acceptableFraction, dctMethod, name)
    }
    
    fun decodeBmp(contents: Output, channels: Long = 0L, name: String = "DecodeBmp"): Output {
      return gen_image_ops.decodeBmp(contents, channels, name)
    }
    
    fun decodeGif(contents: Output, name: String = "DecodeGif"): Output {
      return gen_image_ops.decodeGif(contents, name)
    }
    
    fun decodeJpeg(contents: Output, channels: Long = 0L, ratio: Long = 1L, fancyUpscaling: Boolean = true, tryRecoverTruncated: Boolean = false, acceptableFraction: Float = 1.0f, dctMethod: String = "", name: String = "DecodeJpeg"): Output {
      return gen_image_ops.decodeJpeg(contents, channels, ratio, fancyUpscaling, tryRecoverTruncated, acceptableFraction, dctMethod, name)
    }
    
    fun decodePng(contents: Output, channels: Long = 0L, dtype: DataType<*> = UINT8, name: String = "DecodePng"): Output {
      return gen_image_ops.decodePng(contents, channels, dtype, name)
    }
    
    fun drawBoundingBoxes(images: Output, boxes: Output, name: String = "DrawBoundingBoxes"): Output {
      return gen_image_ops.drawBoundingBoxes(images, boxes, name)
    }
    
    fun encodeJpeg(image: Output, format: String = "", quality: Long = 95L, progressive: Boolean = false, optimizeSize: Boolean = false, chromaDownsampling: Boolean = true, densityUnit: String = "in", xDensity: Long = 300L, yDensity: Long = 300L, xmpMetadata: String = "", name: String = "EncodeJpeg"): Output {
      return gen_image_ops.encodeJpeg(image, format, quality, progressive, optimizeSize, chromaDownsampling, densityUnit, xDensity, yDensity, xmpMetadata, name)
    }
    
    fun encodePng(image: Output, compression: Long = -1L, name: String = "EncodePng"): Output {
      return gen_image_ops.encodePng(image, compression, name)
    }
    
    fun extractGlimpse(input: Output, size: Output, offsets: Output, centered: Boolean = true, normalized: Boolean = true, uniformNoise: Boolean = true, name: String = "ExtractGlimpse"): Output {
      return gen_image_ops.extractGlimpse(input, size, offsets, centered, normalized, uniformNoise, name)
    }
    
    fun extractJpegShape(contents: Output, outputType: DataType<*> = INT32, name: String = "ExtractJpegShape"): Output {
      return gen_image_ops.extractJpegShape(contents, outputType, name)
    }
    
    fun hSVToRGB(images: Output, name: String = "HSVToRGB"): Output {
      return gen_image_ops.hSVToRGB(images, name)
    }
    
    fun nonMaxSuppression(boxes: Output, scores: Output, maxOutputSize: Output, iouThreshold: Float = 0.5f, name: String = "NonMaxSuppression"): Output {
      return gen_image_ops.nonMaxSuppression(boxes, scores, maxOutputSize, iouThreshold, name)
    }
    
    fun nonMaxSuppressionV2(boxes: Output, scores: Output, maxOutputSize: Output, iouThreshold: Output, name: String = "NonMaxSuppressionV2"): Output {
      return gen_image_ops.nonMaxSuppressionV2(boxes, scores, maxOutputSize, iouThreshold, name)
    }
    
    fun nonMaxSuppressionV3(boxes: Output, scores: Output, maxOutputSize: Output, iouThreshold: Output, scoreThreshold: Output, name: String = "NonMaxSuppressionV3"): Output {
      return gen_image_ops.nonMaxSuppressionV3(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, name)
    }
    
    fun nonMaxSuppressionWithOverlaps(overlaps: Output, scores: Output, maxOutputSize: Output, overlapThreshold: Output, scoreThreshold: Output, name: String = "NonMaxSuppressionWithOverlaps"): Output {
      return gen_image_ops.nonMaxSuppressionWithOverlaps(overlaps, scores, maxOutputSize, overlapThreshold, scoreThreshold, name)
    }
    
    fun quantizedResizeBilinear(images: Output, size: Output, min: Output, max: Output, alignCorners: Boolean = false, name: String = "QuantizedResizeBilinear"): List<Output> {
      return gen_image_ops.quantizedResizeBilinear(images, size, min, max, alignCorners, name)
    }
    
    fun rGBToHSV(images: Output, name: String = "RGBToHSV"): Output {
      return gen_image_ops.rGBToHSV(images, name)
    }
    
    fun randomCrop(image: Output, size: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomCrop"): Output {
      return gen_image_ops.randomCrop(image, size, seed, seed2, name)
    }
    
    fun resizeArea(images: Output, size: Output, alignCorners: Boolean = false, name: String = "ResizeArea"): Output {
      return gen_image_ops.resizeArea(images, size, alignCorners, name)
    }
    
    fun resizeBicubic(images: Output, size: Output, alignCorners: Boolean = false, name: String = "ResizeBicubic"): Output {
      return gen_image_ops.resizeBicubic(images, size, alignCorners, name)
    }
    
    fun resizeBicubicGrad(grads: Output, originalImage: Output, alignCorners: Boolean = false, name: String = "ResizeBicubicGrad"): Output {
      return gen_image_ops.resizeBicubicGrad(grads, originalImage, alignCorners, name)
    }
    
    fun resizeBilinear(images: Output, size: Output, alignCorners: Boolean = false, name: String = "ResizeBilinear"): Output {
      return gen_image_ops.resizeBilinear(images, size, alignCorners, name)
    }
    
    fun resizeBilinearGrad(grads: Output, originalImage: Output, alignCorners: Boolean = false, name: String = "ResizeBilinearGrad"): Output {
      return gen_image_ops.resizeBilinearGrad(grads, originalImage, alignCorners, name)
    }
    
    fun resizeNearestNeighbor(images: Output, size: Output, alignCorners: Boolean = false, name: String = "ResizeNearestNeighbor"): Output {
      return gen_image_ops.resizeNearestNeighbor(images, size, alignCorners, name)
    }
    
    fun resizeNearestNeighborGrad(grads: Output, size: Output, alignCorners: Boolean = false, name: String = "ResizeNearestNeighborGrad"): Output {
      return gen_image_ops.resizeNearestNeighborGrad(grads, size, alignCorners, name)
    }
    
    fun sampleDistortedBoundingBox(imageSize: Output, boundingBoxes: Output, seed: Long = 0L, seed2: Long = 0L, minObjectCovered: Float = 0.1f, aspectRatioRange: Array<Float> = arrayOf(0.75f, 1.33f), areaRange: Array<Float> = arrayOf(0.05f, 1.0f), maxAttempts: Long = 100L, useImageIfNoBoundingBoxes: Boolean = false, name: String = "SampleDistortedBoundingBox"): List<Output> {
      return gen_image_ops.sampleDistortedBoundingBox(imageSize, boundingBoxes, seed, seed2, minObjectCovered, aspectRatioRange, areaRange, maxAttempts, useImageIfNoBoundingBoxes, name)
    }
    
    fun sampleDistortedBoundingBoxV2(imageSize: Output, boundingBoxes: Output, minObjectCovered: Output, seed: Long = 0L, seed2: Long = 0L, aspectRatioRange: Array<Float> = arrayOf(0.75f, 1.33f), areaRange: Array<Float> = arrayOf(0.05f, 1.0f), maxAttempts: Long = 100L, useImageIfNoBoundingBoxes: Boolean = false, name: String = "SampleDistortedBoundingBoxV2"): List<Output> {
      return gen_image_ops.sampleDistortedBoundingBoxV2(imageSize, boundingBoxes, minObjectCovered, seed, seed2, aspectRatioRange, areaRange, maxAttempts, useImageIfNoBoundingBoxes, name)
    }
  }
}