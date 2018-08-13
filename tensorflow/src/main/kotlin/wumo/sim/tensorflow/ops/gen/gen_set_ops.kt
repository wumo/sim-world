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

interface gen_set_ops {
  fun _denseToDenseSetOperation(set1: Output, set2: Output, set_operation: String, validate_indices: Boolean = true, name: String = "DenseToDenseSetOperation") = run {
    buildOpTensors("DenseToDenseSetOperation", name) {
      addInput(set1, false)
      addInput(set2, false)
      attr("set_operation", set_operation)
      attr("validate_indices", validate_indices)
    }
  }
  
  fun _denseToSparseSetOperation(set1: Output, set2_indices: Output, set2_values: Output, set2_shape: Output, set_operation: String, validate_indices: Boolean = true, name: String = "DenseToSparseSetOperation") = run {
    buildOpTensors("DenseToSparseSetOperation", name) {
      addInput(set1, false)
      addInput(set2_indices, false)
      addInput(set2_values, false)
      addInput(set2_shape, false)
      attr("set_operation", set_operation)
      attr("validate_indices", validate_indices)
    }
  }
  
  fun _setSize(set_indices: Output, set_values: Output, set_shape: Output, validate_indices: Boolean = true, name: String = "SetSize") = run {
    buildOpTensor("SetSize", name) {
      addInput(set_indices, false)
      addInput(set_values, false)
      addInput(set_shape, false)
      attr("validate_indices", validate_indices)
    }
  }
  
  fun _sparseToSparseSetOperation(set1_indices: Output, set1_values: Output, set1_shape: Output, set2_indices: Output, set2_values: Output, set2_shape: Output, set_operation: String, validate_indices: Boolean = true, name: String = "SparseToSparseSetOperation") = run {
    buildOpTensors("SparseToSparseSetOperation", name) {
      addInput(set1_indices, false)
      addInput(set1_values, false)
      addInput(set1_shape, false)
      addInput(set2_indices, false)
      addInput(set2_values, false)
      addInput(set2_shape, false)
      attr("set_operation", set_operation)
      attr("validate_indices", validate_indices)
    }
  }
}