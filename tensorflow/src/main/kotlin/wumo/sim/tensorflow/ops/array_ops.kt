package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.core.InvalidIndexerException
import wumo.sim.tensorflow.ops.gen.gen_array_ops
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape

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
                    shrink_axis_mask,
                    new_axis_mask,
                    ellipsis_mask,
                    tf.currentNameScope)
  }
}

fun Output.slice(start: Any, end: Any? = null, step: Any = 1): Output =
    get(Slice(start, end, step))

operator fun Output.get(slice_spec: Any): Output =
    get(Index(slice_spec))

object array_ops {
  interface API : gen_array_ops {
    fun concat(values: List<Output>, axis: Output, name: String = "ConcatV2"): Output =
        if (values.size == 1)
          tf.identity(values[0], name = name)
        else
          tf.concatV2(values, axis, name)
    
    fun gather(params: Output, indices: Output, axis: Int = 0, name: String = "GatherV2"): Output {
      if (axis == 0) {
      }
      //TODO detect resource variables
      return super.gatherV2(params, indices, tf.const(axis), name)
    }
    
    fun <T : OutputLike> identity(data: T, name: String): Output {
      return when (data) {
        is Output -> {
          if (data.dataType.isRefType)
            super.refIdentity(data, name)
          else
            super.identity(data, name)
        }
        is IndexedSlices -> TODO()
        is SparseOutput -> TODO()
        else -> TODO()
      }
    }
    
    fun oneHot(indices: Output, depth: Output, on_value: Output = tf.const(1f, "on_value"),
               off_value: Output = tf.const(0f, "off_value"), axis: Long = -1L, name: String = "OneHot") =
        super._oneHot(indices, depth, on_value, off_value, axis, name)
    
    fun ones(shape: Shape, dtype: DataType<*> = FLOAT, name: String = "ones"): Output =
        tf.nameScope(name) {
          if (shape.numElements() < 1000)
            tf.const(shape, dtype, 1, tf.currentNameScope)
          else {
            super.fill(tf.const(shape.asLongArray()),
                       tf.const(dtype, 1), tf.currentNameScope)
          }
        }
    
    fun ones(shape: Output, dtype: DataType<*> = FLOAT, name: String = "ones"): Output =
        tf.nameScope(name) {
          super.fill(shape, tf.const(dtype.baseDataType, 1), tf.currentNameScope)
        }
    
    fun onesLike(x: Output, dtype: DataType<*>? = null, optimize: Boolean = true, name: String = "ones_like"): Output {
      val outptuDataType = dtype ?: x.dataType
      val onesShape = shape(x, optimize = optimize)
      return ones(onesShape, outptuDataType, name = name)
    }
    
    fun placeholder(shape: Shape = Shape(),
                    dtype: DataType<*> = FLOAT, name: String = "Placeholder"): Output =
        super.placeholder(dtype, shape, name)
    
    fun rank(input: Output, name: String = "Rank", optimize: Boolean = true): Output {
      //TODO SparseOutput
      val input_shape = input.shape
      if (optimize && input_shape.isFullyDefined)
        return tf.const(input_shape.rank, name)
      return super.rank(input, name)
    }
    
    /**
     * Returns the shape of a tensor.
     *
     *
     */
    fun shape(input: OutputLike, out_type: DataType<*> = INT32, name: String = "Shape", optimize: Boolean = true): Output {
      return when (input) {
        is SparseOutput -> tf.cast(input.denseShape!!, out_type)
        is Output -> {
          val input_shape = input.shape
          if (optimize && input_shape.isFullyDefined)
            tf.const(input_shape.asIntArray()!!, name)
          else
            super.shape(input, out_type, name)
        }
        else -> TODO()
      }
    }
    
    fun split(input: Output, splitSizes: Output, axis: Output = tf.const(0), name: String = "split"): List<Output> {
      val splitSizesShape = splitSizes.shape
      if (splitSizesShape.isUnknown)
        throw InvalidArgumentException("Cannot infer the number of splits from the shape '$splitSizesShape'.")
      if (splitSizesShape.rank == 0 && splitSizes.dataType.isInteger)
        return super._split(axis, input, splitSizesShape[0].toLong(), name)
      return super.splitV(input, splitSizes, axis, splitSizesShape[0].toLong(), name)
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
          super.pack(values.map {
            when (it) {
              is Output -> it
              is Int -> tf.const(it)
              else -> error("not supported$it")
            }
          }, axis.toLong(), name)
      }
      TODO()
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
      return super.unpack(input, number.toLong(), axis.toLong(), name)
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
    
    fun zerosLike(x: Output, dtype: DataType<*>? = null, optimize: Boolean = true, name: String = "zeros_like") =
        when {
          optimize && x.shape.isFullyDefined && x.dataType != VARIANT ->
            zeros(x.shape, dtype = dtype ?: x.dataType, name = name)
          dtype != null && dtype != x.dataType && dtype != VARIANT ->
            zeros(shape(x, optimize = optimize), dtype = dtype, name = name)
          else -> super.zerosLike(x, name)
        }
    
    fun zeros(shape: Output, dtype: DataType<*> = FLOAT, name: String = "Ones"): Output =
        tf.nameScope(name) {
          val zero = when (dtype) {
            STRING -> ""
            else -> 0
          }
          super.fill(shape, tf.const(dtype, zero), tf.currentNameScope)
        }
    
    fun zeros(shape: Shape, dtype: DataType<*> = FLOAT, name: String = "Ones"): Output =
        tf.nameScope(name) {
          val zero = when (dtype) {
            STRING -> ""
            else -> 0
          }
          if (shape.numElements() < 1000)
            tf.const(shape, dtype, zero, tf.currentNameScope)
          else {
            val shape = super.reshape(tf.const(shape.asLongArray()), tf.const(-1), name)
            super.fill(shape, tf.const(dtype, zero), tf.currentNameScope)
          }
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
