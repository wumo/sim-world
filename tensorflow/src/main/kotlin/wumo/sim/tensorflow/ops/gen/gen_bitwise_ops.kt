/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.ops.Output
import wumo.sim.util.Shape
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray

interface gen_bitwise_ops {
  fun _bitwiseAnd(x: Output, y: Output, name: String = "BitwiseAnd") = run {
    buildOpTensor("BitwiseAnd", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun _bitwiseOr(x: Output, y: Output, name: String = "BitwiseOr") = run {
    buildOpTensor("BitwiseOr", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun _bitwiseXor(x: Output, y: Output, name: String = "BitwiseXor") = run {
    buildOpTensor("BitwiseXor", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun _invert(x: Output, name: String = "Invert") = run {
    buildOpTensor("Invert", name) {
      addInput(x, false)
    }
  }
  
  fun _leftShift(x: Output, y: Output, name: String = "LeftShift") = run {
    buildOpTensor("LeftShift", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun _populationCount(x: Output, name: String = "PopulationCount") = run {
    buildOpTensor("PopulationCount", name) {
      addInput(x, false)
    }
  }
  
  fun _rightShift(x: Output, y: Output, name: String = "RightShift") = run {
    buildOpTensor("RightShift", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
}