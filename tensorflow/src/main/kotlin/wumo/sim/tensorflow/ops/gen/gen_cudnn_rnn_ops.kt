/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType

interface gen_cudnn_rnn_ops {
  fun _cudnnRNN(input: Output, input_h: Output, input_c: Output, params: Output, rnn_mode: String = "lstm", input_mode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, is_training: Boolean = true, name: String = "CudnnRNN") = run {
    buildOpTensors("CudnnRNN", name) {
      addInput(input, false)
      addInput(input_h, false)
      addInput(input_c, false)
      addInput(params, false)
      attr("rnn_mode", rnn_mode)
      attr("input_mode", input_mode)
      attr("direction", direction)
      attr("dropout", dropout)
      attr("seed", seed)
      attr("seed2", seed2)
      attr("is_training", is_training)
    }
  }
  
  fun _cudnnRNNBackprop(input: Output, input_h: Output, input_c: Output, params: Output, output: Output, output_h: Output, output_c: Output, output_backprop: Output, output_h_backprop: Output, output_c_backprop: Output, reserve_space: Output, rnn_mode: String = "lstm", input_mode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNBackprop") = run {
    buildOpTensors("CudnnRNNBackprop", name) {
      addInput(input, false)
      addInput(input_h, false)
      addInput(input_c, false)
      addInput(params, false)
      addInput(output, false)
      addInput(output_h, false)
      addInput(output_c, false)
      addInput(output_backprop, false)
      addInput(output_h_backprop, false)
      addInput(output_c_backprop, false)
      addInput(reserve_space, false)
      attr("rnn_mode", rnn_mode)
      attr("input_mode", input_mode)
      attr("direction", direction)
      attr("dropout", dropout)
      attr("seed", seed)
      attr("seed2", seed2)
    }
  }
  
  fun _cudnnRNNCanonicalToParams(num_layers: Output, num_units: Output, input_size: Output, weights: Array<Output>, biases: Array<Output>, rnn_mode: String = "lstm", input_mode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNCanonicalToParams") = run {
    buildOpTensor("CudnnRNNCanonicalToParams", name) {
      addInput(num_layers, false)
      addInput(num_units, false)
      addInput(input_size, false)
      addInput(weights, false)
      addInput(biases, false)
      attr("rnn_mode", rnn_mode)
      attr("input_mode", input_mode)
      attr("direction", direction)
      attr("dropout", dropout)
      attr("seed", seed)
      attr("seed2", seed2)
    }
  }
  
  fun _cudnnRNNParamsSize(num_layers: Output, num_units: Output, input_size: Output, t: DataType<*>, s: DataType<*>, rnn_mode: String = "lstm", input_mode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNParamsSize") = run {
    buildOpTensor("CudnnRNNParamsSize", name) {
      addInput(num_layers, false)
      addInput(num_units, false)
      addInput(input_size, false)
      attr("T", t)
      attr("S", s)
      attr("rnn_mode", rnn_mode)
      attr("input_mode", input_mode)
      attr("direction", direction)
      attr("dropout", dropout)
      attr("seed", seed)
      attr("seed2", seed2)
    }
  }
  
  fun _cudnnRNNParamsToCanonical(num_layers: Output, num_units: Output, input_size: Output, params: Output, num_params: Long, rnn_mode: String = "lstm", input_mode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNParamsToCanonical") = run {
    buildOpTensors("CudnnRNNParamsToCanonical", name) {
      addInput(num_layers, false)
      addInput(num_units, false)
      addInput(input_size, false)
      addInput(params, false)
      attr("num_params", num_params)
      attr("rnn_mode", rnn_mode)
      attr("input_mode", input_mode)
      attr("direction", direction)
      attr("dropout", dropout)
      attr("seed", seed)
      attr("seed2", seed2)
    }
  }
  
  fun _cudnnRNNBackpropV2(input: Output, input_h: Output, input_c: Output, params: Output, output: Output, output_h: Output, output_c: Output, output_backprop: Output, output_h_backprop: Output, output_c_backprop: Output, reserve_space: Output, host_reserved: Output, rnn_mode: String = "lstm", input_mode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, name: String = "CudnnRNNBackpropV2") = run {
    buildOpTensors("CudnnRNNBackpropV2", name) {
      addInput(input, false)
      addInput(input_h, false)
      addInput(input_c, false)
      addInput(params, false)
      addInput(output, false)
      addInput(output_h, false)
      addInput(output_c, false)
      addInput(output_backprop, false)
      addInput(output_h_backprop, false)
      addInput(output_c_backprop, false)
      addInput(reserve_space, false)
      addInput(host_reserved, false)
      attr("rnn_mode", rnn_mode)
      attr("input_mode", input_mode)
      attr("direction", direction)
      attr("dropout", dropout)
      attr("seed", seed)
      attr("seed2", seed2)
    }
  }
  
  fun _cudnnRNNV2(input: Output, input_h: Output, input_c: Output, params: Output, rnn_mode: String = "lstm", input_mode: String = "linear_input", direction: String = "unidirectional", dropout: Float = 0.0f, seed: Long = 0L, seed2: Long = 0L, is_training: Boolean = true, name: String = "CudnnRNNV2") = run {
    buildOpTensors("CudnnRNNV2", name) {
      addInput(input, false)
      addInput(input_h, false)
      addInput(input_c, false)
      addInput(params, false)
      attr("rnn_mode", rnn_mode)
      attr("input_mode", input_mode)
      attr("direction", direction)
      attr("dropout", dropout)
      attr("seed", seed)
      attr("seed2", seed2)
      attr("is_training", is_training)
    }
  }
}