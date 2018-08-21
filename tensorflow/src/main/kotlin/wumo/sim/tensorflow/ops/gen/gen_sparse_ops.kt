/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.STRING

interface gen_sparse_ops {
  fun _addManySparseToTensorsMap(sparse_indices: Output, sparse_values: Output, sparse_shape: Output, container: String = "", shared_name: String = "", name: String = "AddManySparseToTensorsMap") = run {
    buildOpTensor("AddManySparseToTensorsMap", name) {
      addInput(sparse_indices, false)
      addInput(sparse_values, false)
      addInput(sparse_shape, false)
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun _addSparseToTensorsMap(sparse_indices: Output, sparse_values: Output, sparse_shape: Output, container: String = "", shared_name: String = "", name: String = "AddSparseToTensorsMap") = run {
    buildOpTensor("AddSparseToTensorsMap", name) {
      addInput(sparse_indices, false)
      addInput(sparse_values, false)
      addInput(sparse_shape, false)
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun _deserializeManySparse(serialized_sparse: Output, dtype: DataType<*>, name: String = "DeserializeManySparse") = run {
    buildOpTensors("DeserializeManySparse", name) {
      addInput(serialized_sparse, false)
      attr("dataType", dtype)
    }
  }
  
  fun _deserializeSparse(serialized_sparse: Output, dtype: DataType<*>, name: String = "DeserializeSparse") = run {
    buildOpTensors("DeserializeSparse", name) {
      addInput(serialized_sparse, false)
      attr("dataType", dtype)
    }
  }
  
  fun _serializeManySparse(sparse_indices: Output, sparse_values: Output, sparse_shape: Output, out_type: DataType<*> = STRING, name: String = "SerializeManySparse") = run {
    buildOpTensor("SerializeManySparse", name) {
      addInput(sparse_indices, false)
      addInput(sparse_values, false)
      addInput(sparse_shape, false)
      attr("out_type", out_type)
    }
  }
  
  fun _serializeSparse(sparse_indices: Output, sparse_values: Output, sparse_shape: Output, out_type: DataType<*> = STRING, name: String = "SerializeSparse") = run {
    buildOpTensor("SerializeSparse", name) {
      addInput(sparse_indices, false)
      addInput(sparse_values, false)
      addInput(sparse_shape, false)
      attr("out_type", out_type)
    }
  }
  
  fun _sparseAdd(a_indices: Output, a_values: Output, a_shape: Output, b_indices: Output, b_values: Output, b_shape: Output, thresh: Output, name: String = "SparseAdd") = run {
    buildOpTensors("SparseAdd", name) {
      addInput(a_indices, false)
      addInput(a_values, false)
      addInput(a_shape, false)
      addInput(b_indices, false)
      addInput(b_values, false)
      addInput(b_shape, false)
      addInput(thresh, false)
    }
  }
  
  fun _sparseAddGrad(backprop_val_grad: Output, a_indices: Output, b_indices: Output, sum_indices: Output, name: String = "SparseAddGrad") = run {
    buildOpTensors("SparseAddGrad", name) {
      addInput(backprop_val_grad, false)
      addInput(a_indices, false)
      addInput(b_indices, false)
      addInput(sum_indices, false)
    }
  }
  
  fun _sparseConcat(indices: List<Output>, values: List<Output>, shapes: List<Output>, concat_dim: Long, name: String = "SparseConcat") = run {
    buildOpTensors("SparseConcat", name) {
      addInput(indices, false)
      addInput(values, false)
      addInput(shapes, false)
      attr("concat_dim", concat_dim)
    }
  }
  
  fun _sparseCross(indices: List<Output>, values: Output, shapes: List<Output>, dense_inputs: Output, hashed_output: Boolean, num_buckets: Long, hash_key: Long, out_type: DataType<*>, internal_type: DataType<*>, name: String = "SparseCross") = run {
    buildOpTensors("SparseCross", name) {
      addInput(indices, false)
      addInput(values, false)
      addInput(shapes, false)
      addInput(dense_inputs, false)
      attr("hashed_output", hashed_output)
      attr("num_buckets", num_buckets)
      attr("hash_key", hash_key)
      attr("out_type", out_type)
      attr("internal_type", internal_type)
    }
  }
  
  fun _sparseDenseCwiseAdd(sp_indices: Output, sp_values: Output, sp_shape: Output, dense: Output, name: String = "SparseDenseCwiseAdd") = run {
    buildOpTensor("SparseDenseCwiseAdd", name) {
      addInput(sp_indices, false)
      addInput(sp_values, false)
      addInput(sp_shape, false)
      addInput(dense, false)
    }
  }
  
  fun _sparseDenseCwiseDiv(sp_indices: Output, sp_values: Output, sp_shape: Output, dense: Output, name: String = "SparseDenseCwiseDiv") = run {
    buildOpTensor("SparseDenseCwiseDiv", name) {
      addInput(sp_indices, false)
      addInput(sp_values, false)
      addInput(sp_shape, false)
      addInput(dense, false)
    }
  }
  
  fun _sparseDenseCwiseMul(sp_indices: Output, sp_values: Output, sp_shape: Output, dense: Output, name: String = "SparseDenseCwiseMul") = run {
    buildOpTensor("SparseDenseCwiseMul", name) {
      addInput(sp_indices, false)
      addInput(sp_values, false)
      addInput(sp_shape, false)
      addInput(dense, false)
    }
  }
  
  fun _sparseFillEmptyRows(indices: Output, values: Output, dense_shape: Output, default_value: Output, name: String = "SparseFillEmptyRows") = run {
    buildOpTensors("SparseFillEmptyRows", name) {
      addInput(indices, false)
      addInput(values, false)
      addInput(dense_shape, false)
      addInput(default_value, false)
    }
  }
  
  fun _sparseFillEmptyRowsGrad(reverse_index_map: Output, grad_values: Output, name: String = "SparseFillEmptyRowsGrad") = run {
    buildOpTensors("SparseFillEmptyRowsGrad", name) {
      addInput(reverse_index_map, false)
      addInput(grad_values, false)
    }
  }
  
  fun _sparseReduceMax(input_indices: Output, input_values: Output, input_shape: Output, reduction_axes: Output, keep_dims: Boolean = false, name: String = "SparseReduceMax") = run {
    buildOpTensor("SparseReduceMax", name) {
      addInput(input_indices, false)
      addInput(input_values, false)
      addInput(input_shape, false)
      addInput(reduction_axes, false)
      attr("keep_dims", keep_dims)
    }
  }
  
  fun _sparseReduceMaxSparse(input_indices: Output, input_values: Output, input_shape: Output, reduction_axes: Output, keep_dims: Boolean = false, name: String = "SparseReduceMaxSparse") = run {
    buildOpTensors("SparseReduceMaxSparse", name) {
      addInput(input_indices, false)
      addInput(input_values, false)
      addInput(input_shape, false)
      addInput(reduction_axes, false)
      attr("keep_dims", keep_dims)
    }
  }
  
  fun _sparseReduceSum(input_indices: Output, input_values: Output, input_shape: Output, reduction_axes: Output, keep_dims: Boolean = false, name: String = "SparseReduceSum") = run {
    buildOpTensor("SparseReduceSum", name) {
      addInput(input_indices, false)
      addInput(input_values, false)
      addInput(input_shape, false)
      addInput(reduction_axes, false)
      attr("keep_dims", keep_dims)
    }
  }
  
  fun _sparseReduceSumSparse(input_indices: Output, input_values: Output, input_shape: Output, reduction_axes: Output, keep_dims: Boolean = false, name: String = "SparseReduceSumSparse") = run {
    buildOpTensors("SparseReduceSumSparse", name) {
      addInput(input_indices, false)
      addInput(input_values, false)
      addInput(input_shape, false)
      addInput(reduction_axes, false)
      attr("keep_dims", keep_dims)
    }
  }
  
  fun _sparseReorder(input_indices: Output, input_values: Output, input_shape: Output, name: String = "SparseReorder") = run {
    buildOpTensors("SparseReorder", name) {
      addInput(input_indices, false)
      addInput(input_values, false)
      addInput(input_shape, false)
    }
  }
  
  fun _sparseReshape(input_indices: Output, input_shape: Output, new_shape: Output, name: String = "SparseReshape") = run {
    buildOpTensors("SparseReshape", name) {
      addInput(input_indices, false)
      addInput(input_shape, false)
      addInput(new_shape, false)
    }
  }
  
  fun _sparseSlice(indices: Output, values: Output, shape: Output, start: Output, size: Output, name: String = "SparseSlice") = run {
    buildOpTensors("SparseSlice", name) {
      addInput(indices, false)
      addInput(values, false)
      addInput(shape, false)
      addInput(start, false)
      addInput(size, false)
    }
  }
  
  fun _sparseSliceGrad(backprop_val_grad: Output, input_indices: Output, input_start: Output, output_indices: Output, name: String = "SparseSliceGrad") = run {
    buildOpTensor("SparseSliceGrad", name) {
      addInput(backprop_val_grad, false)
      addInput(input_indices, false)
      addInput(input_start, false)
      addInput(output_indices, false)
    }
  }
  
  fun _sparseSoftmax(sp_indices: Output, sp_values: Output, sp_shape: Output, name: String = "SparseSoftmax") = run {
    buildOpTensor("SparseSoftmax", name) {
      addInput(sp_indices, false)
      addInput(sp_values, false)
      addInput(sp_shape, false)
    }
  }
  
  fun _sparseSparseMaximum(a_indices: Output, a_values: Output, a_shape: Output, b_indices: Output, b_values: Output, b_shape: Output, name: String = "SparseSparseMaximum") = run {
    buildOpTensors("SparseSparseMaximum", name) {
      addInput(a_indices, false)
      addInput(a_values, false)
      addInput(a_shape, false)
      addInput(b_indices, false)
      addInput(b_values, false)
      addInput(b_shape, false)
    }
  }
  
  fun _sparseSparseMinimum(a_indices: Output, a_values: Output, a_shape: Output, b_indices: Output, b_values: Output, b_shape: Output, name: String = "SparseSparseMinimum") = run {
    buildOpTensors("SparseSparseMinimum", name) {
      addInput(a_indices, false)
      addInput(a_values, false)
      addInput(a_shape, false)
      addInput(b_indices, false)
      addInput(b_values, false)
      addInput(b_shape, false)
    }
  }
  
  fun _sparseSplit(split_dim: Output, indices: Output, values: Output, shape: Output, num_split: Long, name: String = "SparseSplit") = run {
    buildOpTensors("SparseSplit", name) {
      addInput(split_dim, false)
      addInput(indices, false)
      addInput(values, false)
      addInput(shape, false)
      attr("num_split", num_split)
    }
  }
  
  fun _sparseTensorDenseAdd(a_indices: Output, a_values: Output, a_shape: Output, b: Output, name: String = "SparseTensorDenseAdd") = run {
    buildOpTensor("SparseTensorDenseAdd", name) {
      addInput(a_indices, false)
      addInput(a_values, false)
      addInput(a_shape, false)
      addInput(b, false)
    }
  }
  
  fun _sparseTensorDenseMatMul(a_indices: Output, a_values: Output, a_shape: Output, b: Output, adjoint_a: Boolean = false, adjoint_b: Boolean = false, name: String = "SparseTensorDenseMatMul") = run {
    buildOpTensor("SparseTensorDenseMatMul", name) {
      addInput(a_indices, false)
      addInput(a_values, false)
      addInput(a_shape, false)
      addInput(b, false)
      attr("adjoint_a", adjoint_a)
      attr("adjoint_b", adjoint_b)
    }
  }
  
  fun _sparseToDense(sparse_indices: Output, output_shape: Output, sparse_values: Output, default_value: Output, validate_indices: Boolean = true, name: String = "SparseToDense") = run {
    buildOpTensor("SparseToDense", name) {
      addInput(sparse_indices, false)
      addInput(output_shape, false)
      addInput(sparse_values, false)
      addInput(default_value, false)
      attr("validate_indices", validate_indices)
    }
  }
  
  fun _takeManySparseFromTensorsMap(sparse_handles: Output, dtype: DataType<*>, container: String = "", shared_name: String = "", name: String = "TakeManySparseFromTensorsMap") = run {
    buildOpTensors("TakeManySparseFromTensorsMap", name) {
      addInput(sparse_handles, false)
      attr("dataType", dtype)
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
}