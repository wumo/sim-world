package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_sparse_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.STRING

object sparse_ops {
  interface API {
    fun addManySparseToTensorsMap(sparseIndices: Output, sparseValues: Output, sparseShape: Output, container: String = "", sharedName: String = "", name: String = "AddManySparseToTensorsMap"): Output {
      return gen_sparse_ops.addManySparseToTensorsMap(sparseIndices, sparseValues, sparseShape, container, sharedName, name)
    }
    
    fun addSparseToTensorsMap(sparseIndices: Output, sparseValues: Output, sparseShape: Output, container: String = "", sharedName: String = "", name: String = "AddSparseToTensorsMap"): Output {
      return gen_sparse_ops.addSparseToTensorsMap(sparseIndices, sparseValues, sparseShape, container, sharedName, name)
    }
    
    fun deserializeManySparse(serializedSparse: Output, dtype: DataType<*>, name: String = "DeserializeManySparse"): List<Output> {
      return gen_sparse_ops.deserializeManySparse(serializedSparse, dtype, name)
    }
    
    fun deserializeSparse(serializedSparse: Output, dtype: DataType<*>, name: String = "DeserializeSparse"): List<Output> {
      return gen_sparse_ops.deserializeSparse(serializedSparse, dtype, name)
    }
    
    fun serializeManySparse(sparseIndices: Output, sparseValues: Output, sparseShape: Output, outType: DataType<*> = STRING, name: String = "SerializeManySparse"): Output {
      return gen_sparse_ops.serializeManySparse(sparseIndices, sparseValues, sparseShape, outType, name)
    }
    
    fun serializeSparse(sparseIndices: Output, sparseValues: Output, sparseShape: Output, outType: DataType<*> = STRING, name: String = "SerializeSparse"): Output {
      return gen_sparse_ops.serializeSparse(sparseIndices, sparseValues, sparseShape, outType, name)
    }
    
    fun sparseAdd(aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output, thresh: Output, name: String = "SparseAdd"): List<Output> {
      return gen_sparse_ops.sparseAdd(aIndices, aValues, aShape, bIndices, bValues, bShape, thresh, name)
    }
    
    fun sparseAddGrad(backpropValGrad: Output, aIndices: Output, bIndices: Output, sumIndices: Output, name: String = "SparseAddGrad"): List<Output> {
      return gen_sparse_ops.sparseAddGrad(backpropValGrad, aIndices, bIndices, sumIndices, name)
    }
    
    fun sparseConcat(indices: List<Output>, values: List<Output>, shapes: List<Output>, concatDim: Long, name: String = "SparseConcat"): List<Output> {
      return gen_sparse_ops.sparseConcat(indices, values, shapes, concatDim, name)
    }
    
    fun sparseCross(indices: List<Output>, values: Output, shapes: List<Output>, denseInputs: Output, hashedOutput: Boolean, numBuckets: Long, hashKey: Long, outType: DataType<*>, internalType: DataType<*>, name: String = "SparseCross"): List<Output> {
      return gen_sparse_ops.sparseCross(indices, values, shapes, denseInputs, hashedOutput, numBuckets, hashKey, outType, internalType, name)
    }
    
    fun sparseDenseCwiseAdd(spIndices: Output, spValues: Output, spShape: Output, dense: Output, name: String = "SparseDenseCwiseAdd"): Output {
      return gen_sparse_ops.sparseDenseCwiseAdd(spIndices, spValues, spShape, dense, name)
    }
    
    fun sparseDenseCwiseDiv(spIndices: Output, spValues: Output, spShape: Output, dense: Output, name: String = "SparseDenseCwiseDiv"): Output {
      return gen_sparse_ops.sparseDenseCwiseDiv(spIndices, spValues, spShape, dense, name)
    }
    
    fun sparseDenseCwiseMul(spIndices: Output, spValues: Output, spShape: Output, dense: Output, name: String = "SparseDenseCwiseMul"): Output {
      return gen_sparse_ops.sparseDenseCwiseMul(spIndices, spValues, spShape, dense, name)
    }
    
    fun sparseFillEmptyRows(indices: Output, values: Output, denseShape: Output, defaultValue: Output, name: String = "SparseFillEmptyRows"): List<Output> {
      return gen_sparse_ops.sparseFillEmptyRows(indices, values, denseShape, defaultValue, name)
    }
    
    fun sparseFillEmptyRowsGrad(reverseIndexMap: Output, gradValues: Output, name: String = "SparseFillEmptyRowsGrad"): List<Output> {
      return gen_sparse_ops.sparseFillEmptyRowsGrad(reverseIndexMap, gradValues, name)
    }
    
    fun sparseReduceMax(inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Boolean = false, name: String = "SparseReduceMax"): Output {
      return gen_sparse_ops.sparseReduceMax(inputIndices, inputValues, inputShape, reductionAxes, keepDims, name)
    }
    
    fun sparseReduceMaxSparse(inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Boolean = false, name: String = "SparseReduceMaxSparse"): List<Output> {
      return gen_sparse_ops.sparseReduceMaxSparse(inputIndices, inputValues, inputShape, reductionAxes, keepDims, name)
    }
    
    fun sparseReduceSum(inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Boolean = false, name: String = "SparseReduceSum"): Output {
      return gen_sparse_ops.sparseReduceSum(inputIndices, inputValues, inputShape, reductionAxes, keepDims, name)
    }
    
    fun sparseReduceSumSparse(inputIndices: Output, inputValues: Output, inputShape: Output, reductionAxes: Output, keepDims: Boolean = false, name: String = "SparseReduceSumSparse"): List<Output> {
      return gen_sparse_ops.sparseReduceSumSparse(inputIndices, inputValues, inputShape, reductionAxes, keepDims, name)
    }
    
    fun sparseReorder(inputIndices: Output, inputValues: Output, inputShape: Output, name: String = "SparseReorder"): List<Output> {
      return gen_sparse_ops.sparseReorder(inputIndices, inputValues, inputShape, name)
    }
    
    fun sparseReshape(inputIndices: Output, inputShape: Output, newShape: Output, name: String = "SparseReshape"): List<Output> {
      return gen_sparse_ops.sparseReshape(inputIndices, inputShape, newShape, name)
    }
    
    fun sparseSlice(indices: Output, values: Output, shape: Output, start: Output, size: Output, name: String = "SparseSlice"): List<Output> {
      return gen_sparse_ops.sparseSlice(indices, values, shape, start, size, name)
    }
    
    fun sparseSliceGrad(backpropValGrad: Output, inputIndices: Output, inputStart: Output, outputIndices: Output, name: String = "SparseSliceGrad"): Output {
      return gen_sparse_ops.sparseSliceGrad(backpropValGrad, inputIndices, inputStart, outputIndices, name)
    }
    
    fun sparseSoftmax(spIndices: Output, spValues: Output, spShape: Output, name: String = "SparseSoftmax"): Output {
      return gen_sparse_ops.sparseSoftmax(spIndices, spValues, spShape, name)
    }
    
    fun sparseSparseMaximum(aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output, name: String = "SparseSparseMaximum"): List<Output> {
      return gen_sparse_ops.sparseSparseMaximum(aIndices, aValues, aShape, bIndices, bValues, bShape, name)
    }
    
    fun sparseSparseMinimum(aIndices: Output, aValues: Output, aShape: Output, bIndices: Output, bValues: Output, bShape: Output, name: String = "SparseSparseMinimum"): List<Output> {
      return gen_sparse_ops.sparseSparseMinimum(aIndices, aValues, aShape, bIndices, bValues, bShape, name)
    }
    
    fun sparseSplit(splitDim: Output, indices: Output, values: Output, shape: Output, numSplit: Long, name: String = "SparseSplit"): List<Output> {
      return gen_sparse_ops.sparseSplit(splitDim, indices, values, shape, numSplit, name)
    }
    
    fun sparseTensorDenseAdd(aIndices: Output, aValues: Output, aShape: Output, b: Output, name: String = "SparseTensorDenseAdd"): Output {
      return gen_sparse_ops.sparseTensorDenseAdd(aIndices, aValues, aShape, b, name)
    }
    
    fun sparseTensorDenseMatMul(aIndices: Output, aValues: Output, aShape: Output, b: Output, adjointA: Boolean = false, adjointB: Boolean = false, name: String = "SparseTensorDenseMatMul"): Output {
      return gen_sparse_ops.sparseTensorDenseMatMul(aIndices, aValues, aShape, b, adjointA, adjointB, name)
    }
    
    fun sparseToDense(sparseIndices: Output, outputShape: Output, sparseValues: Output, defaultValue: Output, validateIndices: Boolean = true, name: String = "SparseToDense"): Output {
      return gen_sparse_ops.sparseToDense(sparseIndices, outputShape, sparseValues, defaultValue, validateIndices, name)
    }
    
    fun takeManySparseFromTensorsMap(sparseHandles: Output, dtype: DataType<*>, container: String = "", sharedName: String = "", name: String = "TakeManySparseFromTensorsMap"): List<Output> {
      return gen_sparse_ops.takeManySparseFromTensorsMap(sparseHandles, dtype, container, sharedName, name)
    }
  }
}