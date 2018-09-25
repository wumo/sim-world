package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.NullableOutputMaker
import wumo.sim.tensorflow.OutputMaker
import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.core.InvalidIndexerException
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.gen.gen_array_ops
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray

sealed class Indexer
object Ellipsis : Indexer()
object NewAxis : Indexer()
class Index(val index: Any) : Indexer()
data class Slice(val start: Any, val end: Any? = null, val step: Any = 1) : Indexer()

/**
 *
This operation extracts the specified region from the tensor.
The notation is similar to NumPy with the restriction that
currently only support basic indexing. That means that
using a non-scalar tensor as input is not currently allowed.

Some useful examples:

```python
# strip leading and trailing 2 elements
foo = constant([1,2,3,4,5,6])
print(foo[2:-2].eval())  # => [3,4]

# skip every row and reverse every column
foo = constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[::2,::-1].eval())  # => [[3,2,1], [9,8,7]]

# Use scalar tensors as indices on both dimensions
print(foo[constant(0), constant(2)].eval())  # => 3

# Insert another dimension
foo = constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[newaxis, :, :].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[:, newaxis, :].eval()) # => [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
print(foo[:, :, newaxis].eval()) # => [[[1],[2],[3]], [[4],[5],[6]],
[[7],[8],[9]]]

# Ellipses (3 equivalent operations)
foo = constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[newaxis, :, :].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[newaxis, ...].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[newaxis].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
```

Notes:
- `newaxis` is `None` as in NumPy.
- An implicit ellipsis is placed at the end of the `slice_spec`
- NumPy advanced indexing is currently not supported.
 *@return The appropriate slice of "tensor", based on "slice_spec".
 */
operator fun Output.get(vararg indexers: Indexer): Output {
  if (indexers.count { it == Ellipsis } > 1)
    throw InvalidIndexerException("Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
  val begin = MutableList<Any>(indexers.size) { 0 }
  val end = MutableList<Any>(indexers.size) { 0 }
  val strides = MutableList<Any>(indexers.size) { 1 }
  
  var shrink_axis_mask = 0L
  var new_axis_mask = 0L
  var begin_mask = 0L
  var end_mask = 0L
  var ellipsis_mask = 0L
  for ((index, s) in indexers.withIndex()) {
    when (s) {
      is Index -> {
        val i = s.index
        begin[index] = i
        end[index] = when (i) {
          is Int -> i + 1
          is Output -> i + 1
          else -> error("not supported $i")
        }
        strides[index] = 1
        shrink_axis_mask = shrink_axis_mask or (1L shl index)
      }
      is Ellipsis -> ellipsis_mask = ellipsis_mask or (1L shl index)
      is NewAxis -> new_axis_mask = new_axis_mask or (1L shl index)
      is Slice -> {
        val (sliceBegin, sliceEnd, sliceStep) = s
        begin[index] = sliceBegin
        if (sliceEnd != null)
          end[index] = sliceEnd
        else {
          end[index] = 0
          end_mask = end_mask or (1L shl index)
        }
        strides[index] = sliceStep
      }
    }
  }
  return tf.nameScope("strided_slice") {
    val packed_begin = tf.stack(begin)
    val packed_end = tf.stack(end)
    val packed_strides = tf.stack(strides)
    tf.stridedSlice(this@get, packed_begin, packed_end, packed_strides,
                    begin_mask,
                    end_mask,
                    ellipsis_mask,
                    new_axis_mask,
                    shrink_axis_mask,
                    tf.currentNameScope)
  }
}

fun Output.slice(start: Any, end: Any? = null, step: Any = 1): Output =
    get(Slice(start, end, step))

operator fun Output.get(slice_spec: Any): Output =
    get(Index(slice_spec))

object array_ops {
  interface API {
    fun batchMatrixBandPart(input: Output, numLower: Output, numUpper: Output, name: String = "BatchMatrixBandPart"): Output {
      return gen_array_ops.batchMatrixBandPart(input, numLower, numUpper, name)
    }
    
    fun batchMatrixDiag(diagonal: Output, name: String = "BatchMatrixDiag"): Output {
      return gen_array_ops.batchMatrixDiag(diagonal, name)
    }
    
    fun batchMatrixDiagPart(input: Output, name: String = "BatchMatrixDiagPart"): Output {
      return gen_array_ops.batchMatrixDiagPart(input, name)
    }
    
    fun batchMatrixSetDiag(input: Output, diagonal: Output, name: String = "BatchMatrixSetDiag"): Output {
      return gen_array_ops.batchMatrixSetDiag(input, diagonal, name)
    }
    
    fun batchToSpace(input: Output, crops: Output, blockSize: Long, name: String = "BatchToSpace"): Output {
      return gen_array_ops.batchToSpace(input, crops, blockSize, name)
    }
    
    fun batchToSpaceND(input: Output, blockShape: Output, crops: Output, name: String = "BatchToSpaceND"): Output {
      return gen_array_ops.batchToSpaceND(input, blockShape, crops, name)
    }
    
    fun bitcast(input: Output, _type: DataType<*>, name: String = "Bitcast"): Output {
      return gen_array_ops.bitcast(input, _type, name)
    }
    
    fun broadcastArgs(s0: Output, s1: Output, name: String = "BroadcastArgs"): Output {
      return gen_array_ops.broadcastArgs(s0, s1, name)
    }
    
    fun broadcastGradientArgs(s0: Output, s1: Output, name: String = "BroadcastGradientArgs"): List<Output> {
      return gen_array_ops.broadcastGradientArgs(s0, s1, name)
    }
    
    fun broadcastTo(input: Output, shape: Output, name: String = "BroadcastTo"): Output {
      return gen_array_ops.broadcastTo(input, shape, name)
    }
    
    fun checkNumerics(tensor: Output, message: String, name: String = "CheckNumerics"): Output {
      return gen_array_ops.checkNumerics(tensor, message, name)
    }
    
    fun concat(concatDim: Output, values: List<Output>, name: String = "Concat"): Output {
      return gen_array_ops.concat(concatDim, values, name)
    }
    
    fun concatOffset(concatDim: Output, shape: List<Output>, name: String = "ConcatOffset"): List<Output> {
      return gen_array_ops.concatOffset(concatDim, shape, name)
    }
    
    fun concat(values: List<Output>, axis: Output, name: String = "ConcatV2"): Output =
        if (values.size == 1)
          tf.identity(values[0], name = name)
        else
          gen_array_ops.concatV2(values, axis, name)
    
    fun conjugateTranspose(x: Output, perm: Output, name: String = "ConjugateTranspose"): Output {
      return gen_array_ops.conjugateTranspose(x, perm, name)
    }
    
    fun const(value: NDArray<*>, dtype: DataType<*>, name: String = "Const"): Output {
      return gen_array_ops.const(value, dtype, name)
    }
    
    fun debugGradientIdentity(input: Output, name: String = "DebugGradientIdentity"): Output {
      return gen_array_ops.debugGradientIdentity(input, name)
    }
    
    fun debugGradientRefIdentity(input: Output, name: String = "DebugGradientRefIdentity"): Output {
      return gen_array_ops.debugGradientRefIdentity(input, name)
    }
    
    fun deepCopy(x: Output, name: String = "DeepCopy"): Output {
      return gen_array_ops.deepCopy(x, name)
    }
    
    fun depthToSpace(input: Output, blockSize: Long, dataFormat: String = "NHWC", name: String = "DepthToSpace"): Output {
      return gen_array_ops.depthToSpace(input, blockSize, dataFormat, name)
    }
    
    fun dequantize(input: Output, minRange: Output, maxRange: Output, mode: String = "MIN_COMBINED", name: String = "Dequantize"): Output {
      return gen_array_ops.dequantize(input, minRange, maxRange, mode, name)
    }
    
    fun diag(diagonal: Output, name: String = "Diag"): Output {
      return gen_array_ops.diag(diagonal, name)
    }
    
    fun diagPart(input: Output, name: String = "DiagPart"): Output {
      return gen_array_ops.diagPart(input, name)
    }
    
    fun editDistance(hypothesisIndices: Output, hypothesisValues: Output, hypothesisShape: Output, truthIndices: Output, truthValues: Output, truthShape: Output, normalize: Boolean = true, name: String = "EditDistance"): Output {
      return gen_array_ops.editDistance(hypothesisIndices, hypothesisValues, hypothesisShape, truthIndices, truthValues, truthShape, normalize, name)
    }
    
    fun empty(shape: Output, dtype: DataType<*>, init: Boolean = false, name: String = "Empty"): Output {
      return gen_array_ops.empty(shape, dtype, init, name)
    }
    
    fun expandDims(input: Output, dim: Output, name: String = "ExpandDims"): Output {
      return gen_array_ops.expandDims(input, dim, name)
    }
    
    fun extractImagePatches(images: Output, ksizes: Array<Long>, strides: Array<Long>, rates: Array<Long>, padding: String, name: String = "ExtractImagePatches"): Output {
      return gen_array_ops.extractImagePatches(images, ksizes, strides, rates, padding, name)
    }
    
    fun fakeQuantWithMinMaxArgs(inputs: Output, min: Float = -6.0f, max: Float = 6.0f, numBits: Long = 8L, narrowRange: Boolean = false, name: String = "FakeQuantWithMinMaxArgs"): Output {
      return gen_array_ops.fakeQuantWithMinMaxArgs(inputs, min, max, numBits, narrowRange, name)
    }
    
    fun fakeQuantWithMinMaxArgsGradient(gradients: Output, inputs: Output, min: Float = -6.0f, max: Float = 6.0f, numBits: Long = 8L, narrowRange: Boolean = false, name: String = "FakeQuantWithMinMaxArgsGradient"): Output {
      return gen_array_ops.fakeQuantWithMinMaxArgsGradient(gradients, inputs, min, max, numBits, narrowRange, name)
    }
    
    fun fakeQuantWithMinMaxVars(inputs: Output, min: Output, max: Output, numBits: Long = 8L, narrowRange: Boolean = false, name: String = "FakeQuantWithMinMaxVars"): Output {
      return gen_array_ops.fakeQuantWithMinMaxVars(inputs, min, max, numBits, narrowRange, name)
    }
    
    fun fakeQuantWithMinMaxVarsGradient(gradients: Output, inputs: Output, min: Output, max: Output, numBits: Long = 8L, narrowRange: Boolean = false, name: String = "FakeQuantWithMinMaxVarsGradient"): List<Output> {
      return gen_array_ops.fakeQuantWithMinMaxVarsGradient(gradients, inputs, min, max, numBits, narrowRange, name)
    }
    
    fun fakeQuantWithMinMaxVarsPerChannel(inputs: Output, min: Output, max: Output, numBits: Long = 8L, narrowRange: Boolean = false, name: String = "FakeQuantWithMinMaxVarsPerChannel"): Output {
      return gen_array_ops.fakeQuantWithMinMaxVarsPerChannel(inputs, min, max, numBits, narrowRange, name)
    }
    
    fun fakeQuantWithMinMaxVarsPerChannelGradient(gradients: Output, inputs: Output, min: Output, max: Output, numBits: Long = 8L, narrowRange: Boolean = false, name: String = "FakeQuantWithMinMaxVarsPerChannelGradient"): List<Output> {
      return gen_array_ops.fakeQuantWithMinMaxVarsPerChannelGradient(gradients, inputs, min, max, numBits, narrowRange, name)
    }
    
    fun fill(dims: Output, value: Output, name: String = "Fill"): Output {
      return gen_array_ops.fill(dims, value, name)
    }
    
    fun gatherNd(params: Output, indices: Output, name: String = "GatherNd"): Output {
      return gen_array_ops.gatherNd(params, indices, name)
    }
    
    fun gather(params: Output, indices: Output, axis: Int = 0, name: String = "GatherV2"): Output {
      if (axis == 0) {
        gen_array_ops.gatherV2(params, indices, tf.const(axis), name)
      }
      //TODO detect resource variables
      return gen_array_ops.gatherV2(params, indices, tf.const(axis), name)
    }
    
    fun guaranteeConst(input: Output, name: String = "GuaranteeConst"): Output {
      return gen_array_ops.guaranteeConst(input, name)
    }
    
    fun <T : OutputLike> identity(data: T, name: String = "Identity"): Output {
      return when (data) {
        is Output -> {
          gen_array_ops.identity(data, name)
        }
        is IndexedSlices -> TODO()
        is SparseOutput -> TODO()
        else -> TODO()
      }
    }
    
    fun identityN(input: Output, name: String = "IdentityN"): List<Output> {
      return gen_array_ops.identityN(input, name)
    }
    
    fun immutableConst(dtype: DataType<*>, shape: Shape, memoryRegionName: String, name: String = "ImmutableConst"): Output {
      return gen_array_ops.immutableConst(dtype, shape, memoryRegionName, name)
    }
    
    fun inplaceAdd(x: Output, i: Output, v: Output, name: String = "InplaceAdd"): Output {
      return gen_array_ops.inplaceAdd(x, i, v, name)
    }
    
    fun inplaceSub(x: Output, i: Output, v: Output, name: String = "InplaceSub"): Output {
      return gen_array_ops.inplaceSub(x, i, v, name)
    }
    
    fun inplaceUpdate(x: Output, i: Output, v: Output, name: String = "InplaceUpdate"): Output {
      return gen_array_ops.inplaceUpdate(x, i, v, name)
    }
    
    fun invertPermutation(x: Output, name: String = "InvertPermutation"): Output {
      return gen_array_ops.invertPermutation(x, name)
    }
    
    fun listDiff(x: Output, y: Output, outIdx: DataType<*> = INT32, name: String = "ListDiff"): List<Output> {
      return gen_array_ops.listDiff(x, y, outIdx, name)
    }
    
    fun matrixBandPart(input: Output, numLower: Output, numUpper: Output, name: String = "MatrixBandPart"): Output {
      return gen_array_ops.matrixBandPart(input, numLower, numUpper, name)
    }
    
    fun matrixDiag(diagonal: Output, name: String = "MatrixDiag"): Output {
      return gen_array_ops.matrixDiag(diagonal, name)
    }
    
    fun matrixDiagPart(input: Output, name: String = "MatrixDiagPart"): Output {
      return gen_array_ops.matrixDiagPart(input, name)
    }
    
    fun matrixSetDiag(input: Output, diagonal: Output, name: String = "MatrixSetDiag"): Output {
      return gen_array_ops.matrixSetDiag(input, diagonal, name)
    }
    
    fun mirrorPad(input: Output, paddings: Output, mode: String, name: String = "MirrorPad"): Output {
      return gen_array_ops.mirrorPad(input, paddings, mode, name)
    }
    
    fun mirrorPadGrad(input: Output, paddings: Output, mode: String, name: String = "MirrorPadGrad"): Output {
      return gen_array_ops.mirrorPadGrad(input, paddings, mode, name)
    }
    
    fun oneHot(indices: Output,
               depth: Output,
               on_value: Output? = null,
               off_value: Output? = null,
               axis: Long = -1L,
               dataType: DataType<*>? = null,
               name: String = "OneHot"): Output =
        oneHot({ indices }, { depth }, { on_value }, { off_value }, axis, dataType, name)
    
    fun oneHot(indices: OutputMaker,
               depth: OutputMaker,
               on_value: NullableOutputMaker = { null },
               off_value: NullableOutputMaker = { null },
               axis: Long = -1L,
               dataType: DataType<*>? = null,
               name: String = "OneHot"): Output {
      return tf.nameScope(name) {
        val indices = indices("indices")
        val depth = depth("depth")
        val on_value = on_value("on_value")
        val off_value = off_value("off_value")
        val inferredDataType = dataType ?: when {
          on_value != null && off_value != null ->
            DataType.mostPrecise(on_value.dataType, off_value.dataType)
          on_value != null -> on_value.dataType
          off_value != null -> off_value.dataType
          else -> FLOAT
        }
        gen_array_ops.oneHot(
            indices,
            depth,
            on_value?.cast(inferredDataType) ?: tf.const(inferredDataType, 1, "on_value"),
            off_value?.cast(inferredDataType) ?: tf.const(inferredDataType, 0, "off_value"),
            axis,
            tf.currentNameScope
        )
      }
    }
    
    fun ones(shape: Shape, dtype: DataType<*> = FLOAT, name: String = "ones"): Output =
        tf.nameScope(name) {
          if (shape.numElements() < 1000)
            tf.const(shape, dtype, 1, tf.currentNameScope)
          else {
            gen_array_ops.fill(tf.const(shape.asLongArray()!!),
                               tf.const(dtype, 1), tf.currentNameScope)
          }
        }
    
    fun ones(shape: Output, dtype: DataType<*> = FLOAT, name: String = "ones"): Output =
        tf.nameScope(name) {
          gen_array_ops.fill(shape, tf.const(dtype.baseDataType, 1), tf.currentNameScope)
        }
    
    fun onesLike(x: Output, dtype: DataType<*>? = null, optimize: Boolean = true, name: String = "ones_like"): Output {
      val outptuDataType = dtype ?: x.dataType
      val onesShape = shape(x, optimize = optimize)
      return ones(onesShape, outptuDataType, name = name)
    }
    
    fun pad(input: Output, paddings: Output, name: String = "Pad"): Output {
      return gen_array_ops.pad(input, paddings, name)
    }
    
    fun padV2(input: Output, paddings: Output, constantValues: Output, name: String = "PadV2"): Output {
      return gen_array_ops.padV2(input, paddings, constantValues, name)
    }
    
    fun parallelConcat(values: List<Output>, shape: Shape, name: String = "ParallelConcat"): Output {
      return gen_array_ops.parallelConcat(values, shape, name)
    }
    
    fun placeholder(shape: Shape = Shape(),
                    dtype: DataType<*> = FLOAT, name: String = "Placeholder"): Output =
        gen_array_ops.placeholder(dtype, shape, name)
    
    fun placeholderV2(dtype: DataType<*>, shape: Shape, name: String = "PlaceholderV2"): Output {
      return gen_array_ops.placeholderV2(dtype, shape, name)
    }
    
    fun placeholderWithDefault(input: Output, shape: Shape, name: String = "PlaceholderWithDefault"): Output {
      return gen_array_ops.placeholderWithDefault(input, shape, name)
    }
    
    fun preventGradient(input: Output, message: String = "", name: String = "PreventGradient"): Output {
      return gen_array_ops.preventGradient(input, message, name)
    }
    
    fun quantizeAndDequantize(input: Output, signedInput: Boolean = true, numBits: Long = 8L, rangeGiven: Boolean = false, inputMin: Float = 0.0f, inputMax: Float = 0.0f, name: String = "QuantizeAndDequantize"): Output {
      return gen_array_ops.quantizeAndDequantize(input, signedInput, numBits, rangeGiven, inputMin, inputMax, name)
    }
    
    fun quantizeAndDequantizeV2(input: Output, inputMin: Output, inputMax: Output, signedInput: Boolean = true, numBits: Long = 8L, rangeGiven: Boolean = false, name: String = "QuantizeAndDequantizeV2"): Output {
      return gen_array_ops.quantizeAndDequantizeV2(input, inputMin, inputMax, signedInput, numBits, rangeGiven, name)
    }
    
    fun quantizeAndDequantizeV3(input: Output, inputMin: Output, inputMax: Output, numBits: Output, signedInput: Boolean = true, rangeGiven: Boolean = true, name: String = "QuantizeAndDequantizeV3"): Output {
      return gen_array_ops.quantizeAndDequantizeV3(input, inputMin, inputMax, numBits, signedInput, rangeGiven, name)
    }
    
    fun quantizeV2(input: Output, minRange: Output, maxRange: Output, t: DataType<*>, mode: String = "MIN_COMBINED", roundMode: String = "HALF_AWAY_FROM_ZERO", name: String = "QuantizeV2"): List<Output> {
      return gen_array_ops.quantizeV2(input, minRange, maxRange, t, mode, roundMode, name)
    }
    
    fun quantizedConcat(concatDim: Output, values: List<Output>, inputMins: List<Output>, inputMaxes: List<Output>, name: String = "QuantizedConcat"): List<Output> {
      return gen_array_ops.quantizedConcat(concatDim, values, inputMins, inputMaxes, name)
    }
    
    fun quantizedInstanceNorm(x: Output, xMin: Output, xMax: Output, outputRangeGiven: Boolean = false, givenYMin: Float = 0.0f, givenYMax: Float = 0.0f, varianceEpsilon: Float = 1.0E-5f, minSeparation: Float = 0.001f, name: String = "QuantizedInstanceNorm"): List<Output> {
      return gen_array_ops.quantizedInstanceNorm(x, xMin, xMax, outputRangeGiven, givenYMin, givenYMax, varianceEpsilon, minSeparation, name)
    }
    
    fun quantizedReshape(tensor: Output, shape: Output, inputMin: Output, inputMax: Output, name: String = "QuantizedReshape"): List<Output> {
      return gen_array_ops.quantizedReshape(tensor, shape, inputMin, inputMax, name)
    }
    
    fun rank(input: Output, name: String = "Rank", optimize: Boolean = true): Output {
      //TODO SparseOutput
      val input_shape = input.shape
      if (optimize && input_shape.isFullyDefined)
        return tf.const(input_shape.rank, name)
      return gen_array_ops.rank(input, name)
    }
    
    fun refIdentity(input: Output, name: String = "RefIdentity"): Output {
      return gen_array_ops.refIdentity(input, name)
    }
    
    fun requiredSpaceToBatchPaddings(inputShape: Output,
                                     blockShape: Output,
                                     basePaddings: Output? = null,
                                     name: String = "required_space_to_batch_paddings"): Output {
      TODO()
    }
    
    fun reshape(tensor: Output, shape: Output, name: String = "Reshape"): Output {
      return gen_array_ops.reshape(tensor, shape, name)
    }
    
    fun resourceStridedSliceAssign(_ref: Output, begin: Output, end: Output, strides: Output, value: Output, beginMask: Long = 0L, endMask: Long = 0L, ellipsisMask: Long = 0L, newAxisMask: Long = 0L, shrinkAxisMask: Long = 0L, name: String = "ResourceStridedSliceAssign"): Op {
      return gen_array_ops.resourceStridedSliceAssign(_ref, begin, end, strides, value, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, name)
    }
    
    fun reverse(tensor: Output, dims: Output, name: String = "Reverse"): Output {
      return gen_array_ops.reverse(tensor, dims, name)
    }
    
    fun reverseSequence(input: Output, seqLengths: Output, seqDim: Long, batchDim: Long = 0L, name: String = "ReverseSequence"): Output {
      return gen_array_ops.reverseSequence(input, seqLengths, seqDim, batchDim, name)
    }
    
    fun reverseV2(tensor: Output, axis: Output, name: String = "ReverseV2"): Output {
      return gen_array_ops.reverseV2(tensor, axis, name)
    }
    
    fun scatterNd(indices: Output, updates: Output, shape: Output, name: String = "ScatterNd"): Output {
      return gen_array_ops.scatterNd(indices, updates, shape, name)
    }
    
    fun scatterNdNonAliasingAdd(input: Output, indices: Output, updates: Output, name: String = "ScatterNdNonAliasingAdd"): Output {
      return gen_array_ops.scatterNdNonAliasingAdd(input, indices, updates, name)
    }
    
    fun shape(input: OutputLike, outType: DataType<*> = INT32, name: String = "Shape", optimize: Boolean = true): Output {
      return when (input) {
        is SparseOutput -> tf.cast(input.denseShape!!, outType)
        is Output -> {
          val input_shape = input.shape
          if (optimize && input_shape.isFullyDefined)
            tf.const(input_shape.asIntArray()!!, name)
          else
            gen_array_ops.shape(input, outType, name)
        }
        else -> TODO()
      }
    }
    
    fun shapeN(input: List<Output>, outType: DataType<*> = INT32, name: String = "ShapeN"): List<Output> {
      return gen_array_ops.shapeN(input, outType, name)
    }
    
    fun size(input: Output, outType: DataType<*> = INT32, name: String = "Size"): Output {
      return gen_array_ops.size(input, outType, name)
    }
    
    fun slice(input: Output, begin: Output, size: Output, name: String = "Slice"): Output {
      return gen_array_ops.slice(input, begin, size, name)
    }
    
    fun snapshot(input: Output, name: String = "Snapshot"): Output {
      return gen_array_ops.snapshot(input, name)
    }
    
    fun spaceToBatch(input: Output, paddings: Output, blockSize: Long, name: String = "SpaceToBatch"): Output {
      return gen_array_ops.spaceToBatch(input, paddings, blockSize, name)
    }
    
    fun spaceToBatchND(input: Output, blockShape: Output, paddings: Output, name: String = "SpaceToBatchND"): Output {
      return gen_array_ops.spaceToBatchND(input, blockShape, paddings, name)
    }
    
    fun spaceToDepth(input: Output, blockSize: Long, dataFormat: String = "NHWC", name: String = "SpaceToDepth"): Output {
      return gen_array_ops.spaceToDepth(input, blockSize, dataFormat, name)
    }
    
    fun split(input: Output, splitSizes: Output, axis: Output = tf.const(0), name: String = "split"): List<Output> {
      val splitSizesShape = splitSizes.shape
      if (splitSizesShape.isUnknown)
        throw InvalidArgumentException("Cannot infer the number of splits from the shape '$splitSizesShape'.")
      if (splitSizesShape.rank == 0 && splitSizes.dataType.isInteger)
        return gen_array_ops.split(axis, input, splitSizesShape[0].toLong(), name)
      return gen_array_ops.splitV(input, splitSizes, axis, splitSizesShape[0].toLong(), name)
    }
    
    fun splitV(value: Output, sizeSplits: Output, splitDim: Output, numSplit: Long, name: String = "SplitV"): List<Output> {
      return gen_array_ops.splitV(value, sizeSplits, splitDim, numSplit, name)
    }
    
    fun squeeze(input: Output, squeezeDims: Array<Long> = arrayOf(), name: String = "Squeeze"): Output {
      return gen_array_ops.squeeze(input, squeezeDims, name)
    }
    
    /**
     * Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
     * Packs the list of tensors in `values` into a tensor with rank one higher than
     * each tensor in `values`, by packing them along the `axis` dimension.
     * Given a list of length `N` of tensors of shape `(A, B, C)`;
     *
     * if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
     * if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
     * Etc.
     *
     * For example:
     *
     * ```python
     * x = constant([1, 4])
     * y = constant([2, 5])
     * z = constant([3, 6])
     * stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
     * stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
     * ```
     *
     * This is the opposite of unstack.  The numpy equivalent is
     *
     * ```python
     * stack([x, y, z]) = np.stack([x, y, z])
     * ```
     *
     * @param values A list of `Output` objects with the same shape and type.
     * @param axis An `int`. The axis to stack along. Defaults to the first dimension.
     * Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
     * @param name
     * @return A stacked `Output` with the same type as `values`.
     */
    fun stack(values: List<Any>, axis: Long = 0, name: String = "stack"): Output {
      if (axis == 0L) {
        return if (values.all { it is Int })
          tf.const(IntArray(values.size) { values[it] as Int }, name)
        else
          gen_array_ops.pack(values.map {
            when (it) {
              is Output -> it
              is Int -> tf.const(it)
              else -> error("not supported$it")
            }
          }, axis.toLong(), name)
      }
      TODO()
    }
    
    fun stopGradient(input: Output, name: String = "StopGradient"): Output {
      return gen_array_ops.stopGradient(input, name)
    }
    
    fun stridedSlice(input: Output, begin: Output, end: Output, strides: Output, beginMask: Long = 0L, endMask: Long = 0L, ellipsisMask: Long = 0L, newAxisMask: Long = 0L, shrinkAxisMask: Long = 0L, name: String = "StridedSlice"): Output {
      return gen_array_ops.stridedSlice(input, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, name)
    }
    
    fun stridedSliceAssign(_ref: Output, begin: Output, end: Output, strides: Output, value: Output, beginMask: Long = 0L, endMask: Long = 0L, ellipsisMask: Long = 0L, newAxisMask: Long = 0L, shrinkAxisMask: Long = 0L, name: String = "StridedSliceAssign"): Output {
      return gen_array_ops.stridedSliceAssign(_ref, begin, end, strides, value, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, name)
    }
    
    fun stridedSliceGrad(shape: Output, begin: Output, end: Output, strides: Output, dy: Output, beginMask: Long = 0L, endMask: Long = 0L, ellipsisMask: Long = 0L, newAxisMask: Long = 0L, shrinkAxisMask: Long = 0L, name: String = "StridedSliceGrad"): Output {
      return gen_array_ops.stridedSliceGrad(shape, begin, end, strides, dy, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, name)
    }
    
    fun tile(input: Output, multiples: Output, name: String = "Tile"): Output {
      return gen_array_ops.tile(input, multiples, name)
    }
    
    fun tileGrad(input: Output, multiples: Output, name: String = "TileGrad"): Output {
      return gen_array_ops.tileGrad(input, multiples, name)
    }
    
    fun unique(x: Output, outIdx: DataType<*> = INT32, name: String = "Unique"): List<Output> {
      return gen_array_ops.unique(x, outIdx, name)
    }
    
    fun uniqueV2(x: Output, axis: Output, outIdx: DataType<*> = INT32, name: String = "UniqueV2"): List<Output> {
      return gen_array_ops.uniqueV2(x, axis, outIdx, name)
    }
    
    fun uniqueWithCounts(x: Output, outIdx: DataType<*> = INT32, name: String = "UniqueWithCounts"): List<Output> {
      return gen_array_ops.uniqueWithCounts(x, outIdx, name)
    }
    
    fun uniqueWithCountsV2(x: Output, axis: Output, outIdx: DataType<*> = INT32, name: String = "UniqueWithCountsV2"): List<Output> {
      return gen_array_ops.uniqueWithCountsV2(x, axis, outIdx, name)
    }
    
    fun unravelIndex(indices: Output, dims: Output, name: String = "UnravelIndex"): Output {
      return gen_array_ops.unravelIndex(indices, dims, name)
    }
    
    fun unstack(input: Output, num: Int = -1, axis: Int = 0, name: String = "unstack"): List<Output> {
      val number = if (num >= 0) num else {
        val inputShape = input.shape
        val inputShapeRank = inputShape.rank
        if (inputShapeRank != -1 && (axis < -inputShapeRank || axis >= inputShapeRank))
          throw IndexOutOfBoundsException(
              "Provided axis, $axis, is not in [${-inputShapeRank}, $inputShapeRank).")
        inputShape[axis]
      }
      if (number == -1)
        throw IllegalArgumentException("Cannot infer number of tensors to unstack from shape '${input.shape}'.")
      return gen_array_ops.unpack(input, number.toLong(), axis.toLong(), name)
    }
    
    /**
     * Return the elements, either from `x` or `y`, depending on the `condition`.
     *
     * If both `x` and `y` are None, then this operation returns the coordinates of
     * true elements of `condition`.  The coordinates are returned in a 2-D tensor
     * where the first dimension (rows) represents the number of true elements, and
     * the second dimension (columns) represents the coordinates of the true
     * elements. Keep in mind, the shape of the output tensor can vary depending on
     * how many true values there are in input. Indices are output in row-major
     * order.
     *
     * If both non-None, `x` and `y` must have the same shape.
     * The `condition` tensor must be a scalar if `x` and `y` are scalar.
     * If `x` and `y` are vectors of higher rank, then `condition` must be either a
     * vector with size matching the first dimension of `x`, or must have the same
     * shape as `x`.
     *
     * The `condition` tensor acts as a mask that chooses, based on the value at each
     * element, whether the corresponding element / row in the output should be taken
     * from `x` (if true) or `y` (if false).
     *
     * If `condition` is a vector and `x` and `y` are higher rank matrices, then it
     * chooses which row (outer dimension) to copy from `x` and `y`. If `condition`
     * has the same shape as `x` and `y`, then it chooses which element to copy from
     * `x` and `y`.
     */
    fun where(condition: Output, x: Output, y: Output, name: String = "Where") =
        tf.select(condition, x, y, name)
    
    fun where(input: Output, name: String = "Where"): Output {
      return gen_array_ops.where(input, name)
    }
    
    fun zeros(shape: Output, dtype: DataType<*> = FLOAT, name: String = "Ones"): Output =
        tf.nameScope(name) {
          val zero = when (dtype) {
            STRING -> ""
            else -> 0
          }
          gen_array_ops.fill(shape, tf.const(dtype, zero), tf.currentNameScope)
        }
    
    fun zeros(shape: Shape, dtype: DataType<*> = FLOAT, name: String = "Zeros"): Output =
        tf.nameScope(name) {
          val zero = when (dtype) {
            STRING -> ""
            else -> 0
          }
          if (shape.numElements() < 1000)
            tf.const(shape, dtype, zero, tf.currentNameScope)
          else {
            var shape_t = tf.const(shape.asLongArray()!!)
            if (shape_t.shape.rank == -1)
              shape_t = gen_array_ops.reshape(shape_t, tf.const(-1), name)
            gen_array_ops.fill(shape_t, tf.const(dtype, zero), tf.currentNameScope)
          }
        }
    
    fun zerosLike(x: Output, dtype: DataType<*>? = null, optimize: Boolean = true, name: String = "zeros_like") =
        when {
          optimize && x.shape.isFullyDefined && x.dataType != VARIANT ->
            zeros(x.shape, dtype = dtype ?: x.dataType, name = name)
          dtype != null && dtype != x.dataType && dtype != VARIANT ->
            zeros(shape(x, optimize = optimize), dtype = dtype, name = name)
          else -> gen_array_ops.zerosLike(x, name)
        }
    
    class StridedSliceAttrs(var begin_mask_: Int = 0,
                            var end_mask_: Int = 0,
                            var ellipsis_mask_: Int = 0,
                            var new_axis_mask_: Int = 0,
                            var shrink_axis_mask_: Int = 0)
    
    /**Output conversion function that automatically packs arguments.*/
    fun autopack(v: Array<Output>, name: String = "packed"): Output {
      TODO()
    }
  }
}
