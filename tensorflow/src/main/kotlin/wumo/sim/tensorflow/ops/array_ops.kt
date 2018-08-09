package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.gen.*
import wumo.sim.tensorflow.orUse
import wumo.sim.util.Shape
import wumo.sim.tensorflow.ops.gen.oneHot as _oneHot
import wumo.sim.tensorflow.ops.gen.rank as _rank
import wumo.sim.tensorflow.ops.gen.shape as _shape
import wumo.sim.tensorflow.ops.gen.zerosLike as _zerosLike

fun TF.oneHot(indices: Output, depth: Output, on_value: Output = const(1f, "on_value"),
              off_value: Output = const(0f, "off_value"), axis: Long = -1L, name: String = "OneHot") =
    _oneHot(indices, depth, on_value, off_value, axis, name)

inline fun TF.placeholder(shape: Shape = Shape(),
                          dtype: Int = DT_FLOAT, name: String = "Placeholder"): Output =
    placeholder(dtype, shape, name)

fun TF.zerosLike(x: Output, dtype: Int = DT_INVALID, optimize: Boolean = true, name: String = "ZerosLike") =
    when {
      optimize && x.shape.is_fully_defined && x.dtype != DT_VARIANT ->
        zeros(x.shape, dtype = dtype.orUse(x.dtype), name = name)
      dtype != DT_INVALID && dtype != x.dtype && dtype != DT_VARIANT ->
        zeros(shape(x, optimize = optimize), dtype = dtype, name = name)
      else -> _zerosLike(x, name)
    }

fun TF.zeros(shape: Output, dtype: Int = DT_FLOAT, name: String = "Ones"): Output {
  TODO()
}

fun TF.zeros(shape: Shape, dtype: Int = DT_FLOAT, name: String = "Ones"): Output {
  name_scope(name) {
    val zero = when (dtype) {
      DT_STRING -> ""
      else -> 0
    }
    return if (shape.numElements() < 1000)
      const(shape, dtype, zero, ctxNs.scopeName)
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, zero), ctxNs.scopeName)
    }
  }
}

fun TF.ones(shape: Shape, dtype: Int = DT_FLOAT, name: String = "Ones"): Output {
  name_scope(name) {
    return if (shape.numElements() < 1000)
      const(shape, dtype, 1, ctxNs.scopeName)
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, 1), ctxNs.scopeName)
    }
  }
}

/**
 * Returns the shape of a tensor.
 *
 *
 */
fun TF.shape(input: Output, out_type: Int = DT_INT32, name: String = "Shape", optimize: Boolean = true): Output {
  //TODO SparseOutput
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.asIntArray()!!, name)
  return _shape(input, out_type, name)
}

/**
 *
This operation extracts the specified region from the tensor.
The notation is similar to NumPy with the restriction that
currently only support basic indexing. That means that
using a non-scalar tensor as input is not currently allowed.

Some useful examples:

```python
# strip leading and trailing 2 elements
foo = tf.constant([1,2,3,4,5,6])
print(foo[2:-2].eval())  # => [3,4]

# skip every row and reverse every column
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[::2,::-1].eval())  # => [[3,2,1], [9,8,7]]

# Use scalar tensors as indices on both dimensions
print(foo[tf.constant(0), tf.constant(2)].eval())  # => 3

# Insert another dimension
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[tf.newaxis, :, :].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[:, tf.newaxis, :].eval()) # => [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
print(foo[:, :, tf.newaxis].eval()) # => [[[1],[2],[3]], [[4],[5],[6]],
[[7],[8],[9]]]

# Ellipses (3 equivalent operations)
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[tf.newaxis, :, :].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[tf.newaxis, ...].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[tf.newaxis].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
```

Notes:
- `tf.newaxis` is `None` as in NumPy.
- An implicit ellipsis is placed at the end of the `slice_spec`
- NumPy advanced indexing is currently not supported.
 *@return The appropriate slice of "tensor", based on "slice_spec".
 */
operator fun Output.get(vararg slice_spec: Int): Output {
  val begin = mutableListOf<Int>()
  val end = mutableListOf<Int>()
  val strides = mutableListOf<Int>()
  var shrink_axis_mask = 0L
  var new_axis_mask = 0L
  var begin_mask = 0L
  var end_mask = 0L
  var ellipsis_mask = 0L
  for ((index, s) in slice_spec.withIndex()) {
    //TODO other class
    begin += s
    end += s + 1
    strides += 1
    shrink_axis_mask = shrink_axis_mask or (1L shl index)
  }
  tf.name_scope("strided_slice") {
    val packed_begin = tf.stack(begin)
    val packed_end = tf.stack(end)
    val packed_strides = tf.stack(strides)
    return tf.stridedSlice(this@get, packed_begin, packed_end, packed_strides,
                           begin_mask,
                           end_mask,
                           shrink_axis_mask,
                           new_axis_mask,
                           ellipsis_mask,
                           tf.ctxNs.scopeName)
  }
}

class StridedSliceAttrs(var begin_mask_: Int = 0,
                        var end_mask_: Int = 0,
                        var ellipsis_mask_: Int = 0,
                        var new_axis_mask_: Int = 0,
                        var shrink_axis_mask_: Int = 0)

fun TF.gather(params: Output, indices: Output, axis: Int = 0, name: String = "GatherV2"): Output {
  if (axis == 0) {
  }
  //TODO detect resource variables
  return gatherV2(params, indices, const(axis), name)
}

fun TF.rank(input: Output, name: String = "Rank", optimize: Boolean = true): Output {
  //TODO SparseOutput
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.rank, name)
  return _rank(input, name)
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
 * x = tf.constant([1, 4])
 * y = tf.constant([2, 5])
 * z = tf.constant([3, 6])
 * tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
 * tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
 * ```
 *
 * This is the opposite of unstack.  The numpy equivalent is
 *
 * ```python
 * tf.stack([x, y, z]) = np.stack([x, y, z])
 * ```
 *
 * @param values A list of `Output` objects with the same shape and type.
 * @param axis An `int`. The axis to stack along. Defaults to the first dimension.
 * Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
 * @param name
 * @return A stacked `Output` with the same type as `values`.
 */
fun TF.stack(values: Array<Output>, axis: Long = 0L, name: String = "stack"): Output {
  if (axis == 0L) {
    return pack(values, axis, name)
  }
  TODO()
}

fun TF.stack(values: List<Int>, axis: Int = 0, name: String = "stack"): Output {
  if (axis == 0) {
    return const(values.toIntArray(), name)
  }
  TODO()
}

fun TF.stack(values: Collection<Output>, axis: Long = 0L, name: String = "stack") =
    pack(values.toTypedArray(), axis, name)

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
fun TF.where(condition: Output, x: Output, y: Output, name: String = "Where") =
    select(condition, x, y, name)

fun TF.where(condition: Output, name: String = "Where") = _where(condition, name)
/**Output conversion function that automatically packs arguments.*/
fun TF.autopack(v: Array<Output>, name: String = "packed"): Output {
  TODO()
}
