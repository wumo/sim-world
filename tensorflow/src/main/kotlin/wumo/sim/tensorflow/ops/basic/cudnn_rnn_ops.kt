package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_cudnn_rnn_ops
import wumo.sim.tensorflow.types.DataType

object cudnn_rnn_ops {
  interface API {
    fun cudnnRNN(input: Output, inputH: Output, inputC: Output, params: Output, rnnMode: String = "lstm", inputMode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, isTraining: Boolean = true, name: String = "CudnnRNN"): List<Output> {
      return gen_cudnn_rnn_ops.cudnnRNN(input, inputH, inputC, params, rnnMode, inputMode, direction, dropout, seed, seed2, isTraining, name)
    }
    
    fun cudnnRNNBackprop(input: Output, inputH: Output, inputC: Output, params: Output, output: Output, outputH: Output, outputC: Output, outputBackprop: Output, outputHBackprop: Output, outputCBackprop: Output, reserveSpace: Output, rnnMode: String = "lstm", inputMode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNBackprop"): List<Output> {
      return gen_cudnn_rnn_ops.cudnnRNNBackprop(input, inputH, inputC, params, output, outputH, outputC, outputBackprop, outputHBackprop, outputCBackprop, reserveSpace, rnnMode, inputMode, direction, dropout, seed, seed2, name)
    }
    
    fun cudnnRNNBackpropV2(input: Output, inputH: Output, inputC: Output, params: Output, output: Output, outputH: Output, outputC: Output, outputBackprop: Output, outputHBackprop: Output, outputCBackprop: Output, reserveSpace: Output, hostReserved: Output, rnnMode: String = "lstm", inputMode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNBackpropV2"): List<Output> {
      return gen_cudnn_rnn_ops.cudnnRNNBackpropV2(input, inputH, inputC, params, output, outputH, outputC, outputBackprop, outputHBackprop, outputCBackprop, reserveSpace, hostReserved, rnnMode, inputMode, direction, dropout, seed, seed2, name)
    }
    
    fun cudnnRNNCanonicalToParams(numLayers: Output, numUnits: Output, inputSize: Output, weights: List<Output>, biases: List<Output>, rnnMode: String = "lstm", inputMode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNCanonicalToParams"): Output {
      return gen_cudnn_rnn_ops.cudnnRNNCanonicalToParams(numLayers, numUnits, inputSize, weights, biases, rnnMode, inputMode, direction, dropout, seed, seed2, name)
    }
    
    fun cudnnRNNParamsSize(numLayers: Output, numUnits: Output, inputSize: Output, t: DataType<*>, s: DataType<*>, rnnMode: String = "lstm", inputMode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNParamsSize"): Output {
      return gen_cudnn_rnn_ops.cudnnRNNParamsSize(numLayers, numUnits, inputSize, t, s, rnnMode, inputMode, direction, dropout, seed, seed2, name)
    }
    
    fun cudnnRNNParamsToCanonical(numLayers: Output, numUnits: Output, inputSize: Output, params: Output, numParams: Long, rnnMode: String = "lstm", inputMode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNParamsToCanonical"): List<Output> {
      return gen_cudnn_rnn_ops.cudnnRNNParamsToCanonical(numLayers, numUnits, inputSize, params, numParams, rnnMode, inputMode, direction, dropout, seed, seed2, name)
    }
    
    fun cudnnRNNV2(input: Output, inputH: Output, inputC: Output, params: Output, rnnMode: String = "lstm", inputMode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, isTraining: Boolean = true, name: String = "CudnnRNNV2"): List<Output> {
      return gen_cudnn_rnn_ops.cudnnRNNV2(input, inputH, inputC, params, rnnMode, inputMode, direction, dropout, seed, seed2, isTraining, name)
    }
  }
}