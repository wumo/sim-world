package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.contrib.layers.CNNDataFormat
import wumo.sim.tensorflow.contrib.layers.CNNDataFormat.*
import wumo.sim.tensorflow.contrib.layers.ConvPadding
import wumo.sim.tensorflow.contrib.layers.ConvPadding.SAME
import wumo.sim.tensorflow.contrib.layers.ConvPadding.VALID
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_nn_ops
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape
import wumo.sim.util.errorIf
import wumo.sim.util.isCompatibleWith
import wumo.sim.util.ndarray.NDArray

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
    
    fun conv1D(input: Output,
               filters: Output,
               stride: Int,
               padding: String,
               useCudnnOnGpu: Boolean = true,
               dataFormat: CNNDataFormat? = null,
               name: String = "conv1d"): Output =
        tf.nameScope(name) {
          val spatial_start_dim: Int
          val data_format: String
          val strides: Array<Long>
          when (dataFormat) {
            null, NHWC, NWC -> {
              data_format = NHWC.name
              spatial_start_dim = 1
              strides = arrayOf(1L, 1L, stride.toLong(), 1L)
            }
            NCHW, NCW -> {
              data_format = NCHW.name
              spatial_start_dim = 2
              strides = arrayOf(1L, 1L, 1L, stride.toLong())
            }
            else -> error("data_format must be \"NWC\" or \"NCW\".")
          }
          val value = tf.expandDims(input, tf.const(spatial_start_dim))
          val filter = tf.expandDims(filters, tf.const(0))
          val result = gen_nn_ops.conv2D(value,
                                         filter,
                                         strides,
                                         padding,
                                         useCudnnOnGpu,
                                         dataFormat = data_format)
          tf.squeeze(result, arrayOf(spatial_start_dim.toLong()))
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
  
  private fun getStridesAndDilationRate(num_spatial_dims: Int,
                                        strides: List<Int>?,
                                        dilation_rate: List<Int>?)
      : Pair<List<Int>, List<Int>> {
    val dilation_rate = dilation_rate ?: List(num_spatial_dims) { 1 }
    errorIf(dilation_rate.size != num_spatial_dims) {
      "dilation_rate.size=${dilation_rate.size} but should be $num_spatial_dims"
    }
    errorIf(dilation_rate.any { it < 1 }) {
      "all values of dilation_rate must be positive"
    }
    
    val strides = strides ?: List(num_spatial_dims) { 1 }
    errorIf(strides.size != num_spatial_dims) {
      "strides.size=${strides.size} but should be $num_spatial_dims"
    }
    errorIf(strides.any { it < 1 }) {
      "all values of strides must be positive"
    }
    
    errorIf(strides.any { it > 1 } && dilation_rate.any { it > 1 }) {
      "strides > 1 not supported in conjunction with dilation_rate > 1"
    }
    return strides to dilation_rate
  }
  
  class Convolution private constructor(val conv_op: ConvOpFunc) {
    companion object {
      operator fun invoke(input_shape: Shape,
                          fileterShape: Shape,
                          padding: ConvPadding,
                          strides: List<Int>? = null,
                          dilation_rate: List<Int>? = null,
                          name: String? = null,
                          data_format: CNNDataFormat? = null): Convolution {
        var num_total_dims = fileterShape.rank
        if (num_total_dims == -1)
          num_total_dims = input_shape.rank
        errorIf(num_total_dims == -1) {
          "rank of input or filter must be known"
        }
        
        val num_spatial_dims = num_total_dims - 2
        input_shape.withRank(num_spatial_dims + 2)
        fileterShape.withRank(num_spatial_dims + 2)
        val input_channels_dim: Int
        val spatial_dims: IntRange
        if (data_format == null || !data_format.name.startsWith("NC")) {
          input_channels_dim = input_shape[num_spatial_dims + 1]
          spatial_dims = 1 until num_spatial_dims + 1
        } else {
          input_channels_dim = input_shape[1]
          spatial_dims = 2 until num_spatial_dims + 2
        }
        errorIf(!input_channels_dim.isCompatibleWith(fileterShape[num_spatial_dims])) {
          "number of input channels does not match corresponding dimension " +
              "of filter, $input_channels_dim != ${fileterShape[num_spatial_dims]}"
        }
        val (_strides, _dilation_rate) = getStridesAndDilationRate(
            num_spatial_dims, strides, dilation_rate)
        val conv_op = WithSpaceToBatch(
            input_shape,
            dilation_rate = _dilation_rate,
            padding = padding,
            build_op = { _, padding ->
              val _call = NonAtrousConvolution(input_shape,
                                               fileterShape,
                                               padding,
                                               data_format,
                                               _strides,
                                               name!!);
              { input: Output, filter: Output ->
                _call(input, filter)
              }
            },
            filterShape = fileterShape,
            spatial_dims = spatial_dims,
            dataFormat = data_format)
        return Convolution { input, filter ->
          conv_op(input, filter)
        }
      }
    }
    
    operator fun invoke(input: Output, filter: Output) =
        conv_op(input, filter)
  }
  
  class WithSpaceToBatch private constructor(val call: ConvOpFunc) {
    companion object {
      operator fun invoke(input_shape: Shape,
                          dilation_rate: List<Int>,
                          padding: ConvPadding,
                          build_op: (Int, ConvPadding) -> ConvOpFunc,
                          filterShape: Shape,
                          spatial_dims: IntRange?,
                          dataFormat: CNNDataFormat?): WithSpaceToBatch {
        val dilation_rate = tf.const(dilation_rate.toIntArray(), name = "dilation_rate")
        val dilation_rate_shape = dilation_rate.shape
        val rateShape = dilation_rate_shape.withRank(1)
        errorIf(!dilation_rate_shape.isFullyDefined) {
          "rate must have known shape"
        }
        
        val num_spatial_dims = rateShape[0]
        val starting_spatial_dim = if (dataFormat != null
            && dataFormat.name.startsWith("NC")) 2
        else 1
        
        val _spatialDims = spatial_dims ?: starting_spatial_dim until
        num_spatial_dims+starting_spatial_dim
        val originalSpatialDims = _spatialDims.toList()
        val spatialDims = originalSpatialDims.asSequence()
            .mapTo(mutableSetOf()) { it }.asSequence()
            .sorted().toList()
        errorIf(spatialDims != originalSpatialDims || spatialDims.any { it < 1 }) {
          "spatial_dims must be a montonically increasing sequence of positive integers"
        }
        
        val expected_input_rank = if (dataFormat != null
            && dataFormat.name.startsWith("NC")) spatialDims.last()
        else spatialDims.last() + 1
        
        input_shape.withRankAtLeast(expected_input_rank)
        
        val const_rate = constantValue(dilation_rate)
        var rate_or_const_rate: Any = dilation_rate
        if (const_rate != null) {
          const_rate as NDArray<Int>
          rate_or_const_rate = const_rate
          errorIf(const_rate.any { it < 1 }) {
            "dilation_rate must be positive"
          }
          if (const_rate.all { it == 1 }) {
            val call = build_op(num_spatial_dims, padding)
            return WithSpaceToBatch(call)
          }
        }
        TODO()
//        var base_paddings: Output?
//        when (padding) {
//          SAME -> {
//            val fileterShape = filterShape.toOutput(name = "filter_shape")
//            val const_filter_shape = constantValue(fileterShape)
//            if (const_filter_shape != null) {
//              base_paddings = withSpaceToBatchBasePaddings(
//                  const_filter_shape, num_spatial_dims, rate_or_const_rate)
//            } else {
//              base_paddings = null
//            }
//          }
//          VALID -> {
//            val shape = Shape(num_spatial_dims, 2)
//            base_paddings = NDArray(shape, IntArray(shape.numElements()) { 0 })
//          }
//        }
//        val op = build_op(num_spatial_dims, VALID)
//        val withSpaceToBatchCall: ConvOpFunc = { input, filter ->
//          val shape = if (input_shape.rank != -1)
//            Shape(spatialDims.map { input_shape[it] })
//          else Shape()
//          val input_spatial_shape = if (shape.isFullyDefined)
//            shape.toOutput()
//          else {
//            val input_shape_tensor = tf.shape(input)
//            tf.stack(spatialDims.map { input_shape_tensor[it] })
//          }
//          if (base_paddings == null) {
//            val filter_shape = tf.shape(filter)
//            base_paddings = withSpaceToBatchBasePaddings(
//                filterShape, num_spatial_dims, rate_or_const_rate)
//          }
//          val (paddings, crops) = tf.requiredSpaceToBatchPaddings(
//              input_spatial_shape,
//              dilation_rate,
//              base_paddings)
//
//          val dilation_rate = withSpaceToBatchAdjust(dilation_rate, 1, spatialDims)
//          val paddings = withSpaceToBatchAdjust(paddings, 0, spatialDims)
//          val input_converted = tf.spaceToBatchND(input, dilation_rate, paddings)
//
//          val result = op(input_converted, filter)
//          val result_converted = tf.batchToSpaceND(result, dilation_rate, crops)
//
//          if (dataFormat != null && dataFormat.name.startsWith("NC"))
//            if (result_converted.shape[1] == -1) {
//              val output_shape = result_converted.shape.copy()
//              output_shape[1] = filter.shape[-1]
//              result_converted.setShape(output_shape)
//            }
//          result_converted
//        }
//        return WithSpaceToBatch(withSpaceToBatchCall)
      }
    }
    
    operator fun invoke(input: Output, filter: Output): Output =
        call(input, filter)
  }
  
  private fun withSpaceToBatchBasePaddings(filter_shape: Shape,
                                           num_spatial_dims: Int,
                                           rate_or_const_rate: Any): Output {
    val fileter_spatial_shape = filter_shape.slice(0, num_spatial_dims)
    
    TODO()
  }
  
  private fun withSpaceToBatchAdjust() {
  
  }
  
  class NonAtrousConvolution private constructor(val conv_op: ConvOpFunc) {
    companion object {
      operator fun invoke(input_shape: Shape,
                          fileterShape: Shape,
                          padding: ConvPadding,
                          data_format: CNNDataFormat?,
                          strides: List<Int>?,
                          name: String): NonAtrousConvolution {
        
        val filterShape = fileterShape.withRank(input_shape.rank)
        val input_shape = input_shape.withRank(fileterShape.rank)
        errorIf(input_shape.rank == -1) {
          "Rank of convolution must be known"
        }
        errorIf(input_shape.rank < 3 || input_shape.rank > 5) {
          "`input` and `filter` must have rank at least 3 and at most 5"
        }
        val conv_dims = input_shape.rank - 2
        
        var _strides = strides ?: List(conv_dims) { 1 }
        errorIf(_strides.size != conv_dims) {
          "strides.size=${_strides.size} but should be $conv_dims"
        }
        val conv_op: ConvOpFunc
        when (conv_dims) {
          1 -> {
            val data_format = data_format ?: NWC
            val _strides = _strides[0]
            conv_op = { input, filter ->
              tf.conv1D(input, filter, _strides,
                        padding.name,
                        dataFormat = data_format,
                        name = name)
            }
          }
          2 -> {
            _strides = if (data_format == null || data_format == NHWC) {
              listOf(1) + _strides + 1
            } else if (data_format == NCHW)
              listOf(1, 1) + _strides
            else
              error("data_format must be \"NHWC\" or \"NCHW\".")
            val strides = Array(_strides.size) { _strides[it].toLong() }
            conv_op = { input, filter ->
              gen_nn_ops.conv2D(input, filter,
                                strides,
                                padding.name,
                                dataFormat = (data_format ?: NHWC).name,
                                name = name)
            }
          }
          3 -> {
            _strides = if (data_format == null || data_format == NDHWC)
              listOf(1) + _strides + 1
            else if (data_format == NCDHW)
              listOf(1, 1) + _strides
            else
              error("data_format must be \"NDHWC\" or \"NCDHW\". Have: $data_format")
            val strides = Array(_strides.size) { _strides[it].toLong() }
            conv_op = { input, filter ->
              gen_nn_ops.conv3D(input, filter, strides,
                                padding.name,
                                dataFormat = (data_format ?: NDHWC).name,
                                name = name)
            }
          }
          else -> error("Not supported $conv_dims")
        }
        return NonAtrousConvolution(conv_op)
      }
    }
    
    operator fun invoke(input: Output, filter: Output) =
        conv_op(input, filter)
  }
}
typealias ConvOpFunc = (Output, Output) -> Output