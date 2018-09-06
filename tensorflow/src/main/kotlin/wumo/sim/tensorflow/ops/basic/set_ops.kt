package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_set_ops

object set_ops {
  interface API {
    fun denseToDenseSetOperation(set1: Output, set2: Output, setOperation: String, validateIndices: Boolean = true, name: String = "DenseToDenseSetOperation"): List<Output> {
      return gen_set_ops.denseToDenseSetOperation(set1, set2, setOperation, validateIndices, name)
    }
    
    fun denseToSparseSetOperation(set1: Output, set2Indices: Output, set2Values: Output, set2Shape: Output, setOperation: String, validateIndices: Boolean = true, name: String = "DenseToSparseSetOperation"): List<Output> {
      return gen_set_ops.denseToSparseSetOperation(set1, set2Indices, set2Values, set2Shape, setOperation, validateIndices, name)
    }
    
    fun setSize(setIndices: Output, setValues: Output, setShape: Output, validateIndices: Boolean = true, name: String = "SetSize"): Output {
      return gen_set_ops.setSize(setIndices, setValues, setShape, validateIndices, name)
    }
    
    fun sparseToSparseSetOperation(set1Indices: Output, set1Values: Output, set1Shape: Output, set2Indices: Output, set2Values: Output, set2Shape: Output, setOperation: String, validateIndices: Boolean = true, name: String = "SparseToSparseSetOperation"): List<Output> {
      return gen_set_ops.sparseToSparseSetOperation(set1Indices, set1Values, set1Shape, set2Indices, set2Values, set2Shape, setOperation, validateIndices, name)
    }
  }
}