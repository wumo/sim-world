package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_nn_ops
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.a

object nn_ops {
  interface API {
    fun avgPool(value: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", name: String = "AvgPool"): Output {
      return gen_nn_ops.avgPool(value, ksize, strides, padding, dataFormat, name)
    }
    
    fun avgPool3D(input: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", name: String = "AvgPool3D"): Output {
      return gen_nn_ops.avgPool3D(input, ksize, strides, padding, dataFormat, name)
    }
    
    fun avgPool3DGrad(origInputShape: Output, grad: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", name: String = "AvgPool3DGrad"): Output {
      return gen_nn_ops.avgPool3DGrad(origInputShape, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun avgPoolGrad(origInputShape: Output, grad: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", name: String = "AvgPoolGrad"): Output {
      return gen_nn_ops.avgPoolGrad(origInputShape, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun batchNormWithGlobalNormalization(t: Output, m: Output, v: Output, beta: Output, gamma: Output, varianceEpsilon: Float, scaleAfterNormalization: Boolean, name: String = "BatchNormWithGlobalNormalization"): Output {
      return gen_nn_ops.batchNormWithGlobalNormalization(t, m, v, beta, gamma, varianceEpsilon, scaleAfterNormalization, name)
    }
    
    fun batchNormWithGlobalNormalizationGrad(t: Output, m: Output, v: Output, gamma: Output, backprop: Output, varianceEpsilon: Float, scaleAfterNormalization: Boolean, name: String = "BatchNormWithGlobalNormalizationGrad"): List<Output> {
      return gen_nn_ops.batchNormWithGlobalNormalizationGrad(t, m, v, gamma, backprop, varianceEpsilon, scaleAfterNormalization, name)
    }
    
    fun biasAdd(value: Output, bias: Output, dataFormat: String = "NHWC", name: String = "BiasAdd"): Output {
      return gen_nn_ops.biasAdd(value, bias, dataFormat, name)
    }
    
    fun biasAddGrad(outBackprop: Output, dataFormat: String = "NHWC", name: String = "BiasAddGrad"): Output {
      return gen_nn_ops.biasAddGrad(outBackprop, dataFormat, name)
    }
    
    fun biasAddV1(value: Output, bias: Output, name: String = "BiasAddV1"): Output {
      return gen_nn_ops.biasAddV1(value, bias, name)
    }
    
    fun conv2D(input: Output, filter: Output, strides: Array<Long>, padding: String, useCudnnOnGpu: Boolean = true, dataFormat: String = "NHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L), name: String = "Conv2D"): Output {
      return gen_nn_ops.conv2D(input, filter, strides, padding, useCudnnOnGpu, dataFormat, dilations, name)
    }
    
    fun conv2DBackpropFilter(input: Output, filterSizes: Output, outBackprop: Output, strides: Array<Long>, padding: String, useCudnnOnGpu: Boolean = true, dataFormat: String = "NHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L), name: String = "Conv2DBackpropFilter"): Output {
      return gen_nn_ops.conv2DBackpropFilter(input, filterSizes, outBackprop, strides, padding, useCudnnOnGpu, dataFormat, dilations, name)
    }
    
    fun conv2DBackpropInput(inputSizes: Output, filter: Output, outBackprop: Output, strides: Array<Long>, padding: String, useCudnnOnGpu: Boolean = true, dataFormat: String = "NHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L), name: String = "Conv2DBackpropInput"): Output {
      return gen_nn_ops.conv2DBackpropInput(inputSizes, filter, outBackprop, strides, padding, useCudnnOnGpu, dataFormat, dilations, name)
    }
    
    fun conv3D(input: Output, filter: Output, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L, 1L), name: String = "Conv3D"): Output {
      return gen_nn_ops.conv3D(input, filter, strides, padding, dataFormat, dilations, name)
    }
    
    fun conv3DBackpropFilter(input: Output, filter: Output, outBackprop: Output, strides: Array<Long>, padding: String, dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L, 1L), name: String = "Conv3DBackpropFilter"): Output {
      return gen_nn_ops.conv3DBackpropFilter(input, filter, outBackprop, strides, padding, dilations, name)
    }
    
    fun conv3DBackpropFilterV2(input: Output, filterSizes: Output, outBackprop: Output, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L, 1L), name: String = "Conv3DBackpropFilterV2"): Output {
      return gen_nn_ops.conv3DBackpropFilterV2(input, filterSizes, outBackprop, strides, padding, dataFormat, dilations, name)
    }
    
    fun conv3DBackpropInput(input: Output, filter: Output, outBackprop: Output, strides: Array<Long>, padding: String, dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L, 1L), name: String = "Conv3DBackpropInput"): Output {
      return gen_nn_ops.conv3DBackpropInput(input, filter, outBackprop, strides, padding, dilations, name)
    }
    
    fun conv3DBackpropInputV2(inputSizes: Output, filter: Output, outBackprop: Output, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L, 1L), name: String = "Conv3DBackpropInputV2"): Output {
      return gen_nn_ops.conv3DBackpropInputV2(inputSizes, filter, outBackprop, strides, padding, dataFormat, dilations, name)
    }
    
    fun dataFormatDimMap(x: Output, srcFormat: String = "NHWC", dstFormat: String = "NCHW", name: String = "DataFormatDimMap"): Output {
      return gen_nn_ops.dataFormatDimMap(x, srcFormat, dstFormat, name)
    }
    
    fun dataFormatVecPermute(x: Output, srcFormat: String = "NHWC", dstFormat: String = "NCHW", name: String = "DataFormatVecPermute"): Output {
      return gen_nn_ops.dataFormatVecPermute(x, srcFormat, dstFormat, name)
    }
    
    fun depthwiseConv2dNative(input: Output, filter: Output, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L), name: String = "DepthwiseConv2dNative"): Output {
      return gen_nn_ops.depthwiseConv2dNative(input, filter, strides, padding, dataFormat, dilations, name)
    }
    
    fun depthwiseConv2dNativeBackpropFilter(input: Output, filterSizes: Output, outBackprop: Output, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L), name: String = "DepthwiseConv2dNativeBackpropFilter"): Output {
      return gen_nn_ops.depthwiseConv2dNativeBackpropFilter(input, filterSizes, outBackprop, strides, padding, dataFormat, dilations, name)
    }
    
    fun depthwiseConv2dNativeBackpropInput(inputSizes: Output, filter: Output, outBackprop: Output, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L), name: String = "DepthwiseConv2dNativeBackpropInput"): Output {
      return gen_nn_ops.depthwiseConv2dNativeBackpropInput(inputSizes, filter, outBackprop, strides, padding, dataFormat, dilations, name)
    }
    
    fun dilation2D(input: Output, filter: Output, strides: Array<Long>, rates: Array<Long>, padding: String, name: String = "Dilation2D"): Output {
      return gen_nn_ops.dilation2D(input, filter, strides, rates, padding, name)
    }
    
    fun dilation2DBackpropFilter(input: Output, filter: Output, outBackprop: Output, strides: Array<Long>, rates: Array<Long>, padding: String, name: String = "Dilation2DBackpropFilter"): Output {
      return gen_nn_ops.dilation2DBackpropFilter(input, filter, outBackprop, strides, rates, padding, name)
    }
    
    fun dilation2DBackpropInput(input: Output, filter: Output, outBackprop: Output, strides: Array<Long>, rates: Array<Long>, padding: String, name: String = "Dilation2DBackpropInput"): Output {
      return gen_nn_ops.dilation2DBackpropInput(input, filter, outBackprop, strides, rates, padding, name)
    }
    
    fun elu(features: Output, name: String = "Elu"): Output {
      return gen_nn_ops.elu(features, name)
    }
    
    fun eluGrad(gradients: Output, outputs: Output, name: String = "EluGrad"): Output {
      return gen_nn_ops.eluGrad(gradients, outputs, name)
    }
    
    fun fractionalAvgPool(value: Output, poolingRatio: Array<Float>, pseudoRandom: Boolean = false, overlapping: Boolean = false, deterministic: Boolean = false, seed: Long = 0L, seed2: Long = 0L, name: String = "FractionalAvgPool"): List<Output> {
      return gen_nn_ops.fractionalAvgPool(value, poolingRatio, pseudoRandom, overlapping, deterministic, seed, seed2, name)
    }
    
    fun fractionalAvgPoolGrad(origInputTensorShape: Output, outBackprop: Output, rowPoolingSequence: Output, colPoolingSequence: Output, overlapping: Boolean = false, name: String = "FractionalAvgPoolGrad"): Output {
      return gen_nn_ops.fractionalAvgPoolGrad(origInputTensorShape, outBackprop, rowPoolingSequence, colPoolingSequence, overlapping, name)
    }
    
    fun fractionalMaxPool(value: Output, poolingRatio: Array<Float>, pseudoRandom: Boolean = false, overlapping: Boolean = false, deterministic: Boolean = false, seed: Long = 0L, seed2: Long = 0L, name: String = "FractionalMaxPool"): List<Output> {
      return gen_nn_ops.fractionalMaxPool(value, poolingRatio, pseudoRandom, overlapping, deterministic, seed, seed2, name)
    }
    
    fun fractionalMaxPoolGrad(origInput: Output, origOutput: Output, outBackprop: Output, rowPoolingSequence: Output, colPoolingSequence: Output, overlapping: Boolean = false, name: String = "FractionalMaxPoolGrad"): Output {
      return gen_nn_ops.fractionalMaxPoolGrad(origInput, origOutput, outBackprop, rowPoolingSequence, colPoolingSequence, overlapping, name)
    }
    
    fun fusedBatchNorm(x: Output, scale: Output, offset: Output, mean: Output, variance: Output, epsilon: Float = 1.0E-4f, dataFormat: String = "NHWC", isTraining: Boolean = true, name: String = "FusedBatchNorm"): List<Output> {
      return gen_nn_ops.fusedBatchNorm(x, scale, offset, mean, variance, epsilon, dataFormat, isTraining, name)
    }
    
    fun fusedBatchNormGrad(yBackprop: Output, x: Output, scale: Output, reserveSpace1: Output, reserveSpace2: Output, epsilon: Float = 1.0E-4f, dataFormat: String = "NHWC", isTraining: Boolean = true, name: String = "FusedBatchNormGrad"): List<Output> {
      return gen_nn_ops.fusedBatchNormGrad(yBackprop, x, scale, reserveSpace1, reserveSpace2, epsilon, dataFormat, isTraining, name)
    }
    
    fun fusedBatchNormGradV2(yBackprop: Output, x: Output, scale: Output, reserveSpace1: Output, reserveSpace2: Output, epsilon: Float = 1.0E-4f, dataFormat: String = "NHWC", isTraining: Boolean = true, name: String = "FusedBatchNormGradV2"): List<Output> {
      return gen_nn_ops.fusedBatchNormGradV2(yBackprop, x, scale, reserveSpace1, reserveSpace2, epsilon, dataFormat, isTraining, name)
    }
    
    fun fusedBatchNormV2(x: Output, scale: Output, offset: Output, mean: Output, variance: Output, epsilon: Float = 1.0E-4f, dataFormat: String = "NHWC", isTraining: Boolean = true, name: String = "FusedBatchNormV2"): List<Output> {
      return gen_nn_ops.fusedBatchNormV2(x, scale, offset, mean, variance, epsilon, dataFormat, isTraining, name)
    }
    
    fun fusedPadConv2D(input: Output, paddings: Output, filter: Output, mode: String, strides: Array<Long>, padding: String, name: String = "FusedPadConv2D"): Output {
      return gen_nn_ops.fusedPadConv2D(input, paddings, filter, mode, strides, padding, name)
    }
    
    fun fusedResizeAndPadConv2D(input: Output, size: Output, paddings: Output, filter: Output, mode: String, strides: Array<Long>, padding: String, resizeAlignCorners: Boolean = false, name: String = "FusedResizeAndPadConv2D"): Output {
      return gen_nn_ops.fusedResizeAndPadConv2D(input, size, paddings, filter, mode, strides, padding, resizeAlignCorners, name)
    }
    
    fun inTopK(predictions: Output, targets: Output, k: Long, name: String = "InTopK"): Output {
      return gen_nn_ops.inTopK(predictions, targets, k, name)
    }
    
    fun inTopKV2(predictions: Output, targets: Output, k: Output, name: String = "InTopKV2"): Output {
      return gen_nn_ops.inTopKV2(predictions, targets, k, name)
    }
    
    fun l2Loss(t: Output, name: String = "L2Loss"): Output {
      return gen_nn_ops.l2Loss(t, name)
    }
    
    fun lRN(input: Output, depthRadius: Long = 5L, bias: Float = 1.0f, alpha: Float = 1.0f, beta: Float = 0.5f, name: String = "LRN"): Output {
      return gen_nn_ops.lRN(input, depthRadius, bias, alpha, beta, name)
    }
    
    fun lRNGrad(inputGrads: Output, inputImage: Output, outputImage: Output, depthRadius: Long = 5L, bias: Float = 1.0f, alpha: Float = 1.0f, beta: Float = 0.5f, name: String = "LRNGrad"): Output {
      return gen_nn_ops.lRNGrad(inputGrads, inputImage, outputImage, depthRadius, bias, alpha, beta, name)
    }
    
    fun logSoftmax(logits: Output, name: String = "LogSoftmax"): Output {
      return gen_nn_ops.logSoftmax(logits, name)
    }
    
    fun maxPool(input: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", name: String = "MaxPool"): Output {
      return gen_nn_ops.maxPool(input, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPool3D(input: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", name: String = "MaxPool3D"): Output {
      return gen_nn_ops.maxPool3D(input, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPool3DGrad(origInput: Output, origOutput: Output, grad: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", name: String = "MaxPool3DGrad"): Output {
      return gen_nn_ops.maxPool3DGrad(origInput, origOutput, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPool3DGradGrad(origInput: Output, origOutput: Output, grad: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NDHWC", name: String = "MaxPool3DGradGrad"): Output {
      return gen_nn_ops.maxPool3DGradGrad(origInput, origOutput, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPoolGrad(origInput: Output, origOutput: Output, grad: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", name: String = "MaxPoolGrad"): Output {
      return gen_nn_ops.maxPoolGrad(origInput, origOutput, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPoolGradGrad(origInput: Output, origOutput: Output, grad: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, dataFormat: String = "NHWC", name: String = "MaxPoolGradGrad"): Output {
      return gen_nn_ops.maxPoolGradGrad(origInput, origOutput, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPoolGradGradV2(origInput: Output, origOutput: Output, grad: Output, ksize: Output, strides: Output, padding: String, dataFormat: String = "NHWC", name: String = "MaxPoolGradGradV2"): Output {
      return gen_nn_ops.maxPoolGradGradV2(origInput, origOutput, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPoolGradGradWithArgmax(input: Output, grad: Output, argmax: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, name: String = "MaxPoolGradGradWithArgmax"): Output {
      return gen_nn_ops.maxPoolGradGradWithArgmax(input, grad, argmax, ksize, strides, padding, name)
    }
    
    fun maxPoolGradV2(origInput: Output, origOutput: Output, grad: Output, ksize: Output, strides: Output, padding: String, dataFormat: String = "NHWC", name: String = "MaxPoolGradV2"): Output {
      return gen_nn_ops.maxPoolGradV2(origInput, origOutput, grad, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPoolGradWithArgmax(input: Output, grad: Output, argmax: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, name: String = "MaxPoolGradWithArgmax"): Output {
      return gen_nn_ops.maxPoolGradWithArgmax(input, grad, argmax, ksize, strides, padding, name)
    }
    
    fun maxPoolV2(input: Output, ksize: Output, strides: Output, padding: String, dataFormat: String = "NHWC", name: String = "MaxPoolV2"): Output {
      return gen_nn_ops.maxPoolV2(input, ksize, strides, padding, dataFormat, name)
    }
    
    fun maxPoolWithArgmax(input: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, targmax: DataType<*> = INT64, name: String = "MaxPoolWithArgmax"): List<Output> {
      return gen_nn_ops.maxPoolWithArgmax(input, ksize, strides, padding, targmax, name)
    }
    
    fun nthElement(input: Output, n: Output, reverse: Boolean = false, name: String = "NthElement"): Output {
      return gen_nn_ops.nthElement(input, n, reverse, name)
    }
    
    fun quantizedAvgPool(input: Output, minInput: Output, maxInput: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, name: String = "QuantizedAvgPool"): List<Output> {
      return gen_nn_ops.quantizedAvgPool(input, minInput, maxInput, ksize, strides, padding, name)
    }
    
    fun quantizedBatchNormWithGlobalNormalization(t: Output, tMin: Output, tMax: Output, m: Output, mMin: Output, mMax: Output, v: Output, vMin: Output, vMax: Output, beta: Output, betaMin: Output, betaMax: Output, gamma: Output, gammaMin: Output, gammaMax: Output, outType: DataType<*>, varianceEpsilon: Float, scaleAfterNormalization: Boolean, name: String = "QuantizedBatchNormWithGlobalNormalization"): List<Output> {
      return gen_nn_ops.quantizedBatchNormWithGlobalNormalization(t, tMin, tMax, m, mMin, mMax, v, vMin, vMax, beta, betaMin, betaMax, gamma, gammaMin, gammaMax, outType, varianceEpsilon, scaleAfterNormalization, name)
    }
    
    fun quantizedBiasAdd(input: Output, bias: Output, minInput: Output, maxInput: Output, minBias: Output, maxBias: Output, outType: DataType<*>, name: String = "QuantizedBiasAdd"): List<Output> {
      return gen_nn_ops.quantizedBiasAdd(input, bias, minInput, maxInput, minBias, maxBias, outType, name)
    }
    
    fun quantizedConv2D(input: Output, filter: Output, minInput: Output, maxInput: Output, minFilter: Output, maxFilter: Output, strides: Array<Long>, padding: String, outType: DataType<*> = QINT32, dilations: Array<Long> = arrayOf(1L, 1L, 1L, 1L), name: String = "QuantizedConv2D"): List<Output> {
      return gen_nn_ops.quantizedConv2D(input, filter, minInput, maxInput, minFilter, maxFilter, strides, padding, outType, dilations, name)
    }
    
    fun quantizedMaxPool(input: Output, minInput: Output, maxInput: Output, ksize: Array<Long>, strides: Array<Long>, padding: String, name: String = "QuantizedMaxPool"): List<Output> {
      return gen_nn_ops.quantizedMaxPool(input, minInput, maxInput, ksize, strides, padding, name)
    }
    
    fun quantizedRelu(features: Output, minFeatures: Output, maxFeatures: Output, outType: DataType<*> = QUINT8, name: String = "QuantizedRelu"): List<Output> {
      return gen_nn_ops.quantizedRelu(features, minFeatures, maxFeatures, outType, name)
    }
    
    fun quantizedRelu6(features: Output, minFeatures: Output, maxFeatures: Output, outType: DataType<*> = QUINT8, name: String = "QuantizedRelu6"): List<Output> {
      return gen_nn_ops.quantizedRelu6(features, minFeatures, maxFeatures, outType, name)
    }
    
    fun quantizedReluX(features: Output, maxValue: Output, minFeatures: Output, maxFeatures: Output, outType: DataType<*> = QUINT8, name: String = "QuantizedReluX"): List<Output> {
      return gen_nn_ops.quantizedReluX(features, maxValue, minFeatures, maxFeatures, outType, name)
    }
    
    fun relu(features: Output, name: String = "Relu"): Output {
      return gen_nn_ops.relu(features, name)
    }
    
    fun relu6(features: Output, name: String = "Relu6"): Output {
      return gen_nn_ops.relu6(features, name)
    }
    
    fun relu6Grad(gradients: Output, features: Output, name: String = "Relu6Grad"): Output {
      return gen_nn_ops.relu6Grad(gradients, features, name)
    }
    
    fun reluGrad(gradients: Output, features: Output, name: String = "ReluGrad"): Output {
      return gen_nn_ops.reluGrad(gradients, features, name)
    }
    
    fun selu(features: Output, name: String = "Selu"): Output {
      return gen_nn_ops.selu(features, name)
    }
    
    fun seluGrad(gradients: Output, outputs: Output, name: String = "SeluGrad"): Output {
      return gen_nn_ops.seluGrad(gradients, outputs, name)
    }
    
    fun softmax(logits: Output, name: String = "Softmax"): Output {
      return gen_nn_ops.softmax(logits, name)
    }
    
    fun softmaxCrossEntropyWithLogits(features: Output, labels: Output, name: String = "SoftmaxCrossEntropyWithLogits"): List<Output> {
      return gen_nn_ops.softmaxCrossEntropyWithLogits(features, labels, name)
    }
    
    fun softplus(features: Output, name: String = "Softplus"): Output {
      return gen_nn_ops.softplus(features, name)
    }
    
    fun softplusGrad(gradients: Output, features: Output, name: String = "SoftplusGrad"): Output {
      return gen_nn_ops.softplusGrad(gradients, features, name)
    }
    
    fun softsign(features: Output, name: String = "Softsign"): Output {
      return gen_nn_ops.softsign(features, name)
    }
    
    fun softsignGrad(gradients: Output, features: Output, name: String = "SoftsignGrad"): Output {
      return gen_nn_ops.softsignGrad(gradients, features, name)
    }
    
    fun sparseSoftmaxCrossEntropyWithLogits(features: Output, labels: Output, name: String = "SparseSoftmaxCrossEntropyWithLogits"): List<Output> {
      return gen_nn_ops.sparseSoftmaxCrossEntropyWithLogits(features, labels, name)
    }
    
    fun topK(input: Output, k: Long, sorted: Boolean = true, name: String = "TopK"): List<Output> {
      return gen_nn_ops.topK(input, k, sorted, name)
    }
    
    fun topKV2(input: Output, k: Output, sorted: Boolean = true, name: String = "TopKV2"): List<Output> {
      return gen_nn_ops.topKV2(input, k, sorted, name)
    }
    
    fun moments(x: Output,
                axes: LongArray,
                name: String = "moments",
                keep_dims: Boolean = false): List<Output> =
        tf.nameScope(name) {
          //The dynamic range of fp16 is too limited to support the collection of
          //sufficient statistics. As a workaround we simply perform the operations
          //on 32-bit floats before converting the mean and variance back to fp16
          val y = if (x.dataType == FLOAT16) tf.cast(x, FLOAT) else x
          //Compute true mean while keeping the dims for proper broadcasting.
          var mean = tf.mean(y, tf.const(axes), keepDims = true, name = "mean")
          //sample variance, not unbiased variance
          var variance = tf.mean(tf.squaredDifference(y, tf.stopGradient(mean)),
                                 tf.const(axes), keepDims = true, name = "variance")
          if (!keep_dims) {
            mean = tf.squeeze(mean, axes.toTypedArray())
            variance = tf.squeeze(variance, axes.toTypedArray())
          }
          if (x.dataType == FLOAT16)
            listOf(tf.cast(mean, FLOAT16), tf.cast(variance, FLOAT16))
          else listOf(mean, variance)
        }
    
    fun batchNormalization(x: Output,
                           mean: Output, variance: Output,
                           offset: Output?, scale: Output?,
                           variance_epsilon: Float,
                           name: String = "batchnorm"): Output =
        tf.nameScope(name) {
          //    val _variance_epsilon = tf.const(variance_epsilon)
          var inv = tf.rsqrt(variance + variance_epsilon)
          if (scale != null)
            inv *= scale
          x * tf.cast(inv, x.dataType) + tf.cast(
              if (offset != null) offset - mean * inv
              else -mean * inv,
              x.dataType)
        }
  }
}