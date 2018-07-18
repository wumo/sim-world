package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.util.Dimension

fun TF.identity(input: Tensor, name: String = "Identity") =
    unaryOp("Identity", input, name)

fun TF.placeholder(dtype: Int = DT_FLOAT, name: String = "Placeholder"): Tensor {
  val tensor_shape_proto = TensorShapeProto()
  tensor_shape_proto.set_unknown_rank(true)
  val p = g.nodeBuilder("Placeholder", ctx.getUniqueFullName(name))
      .setAttrType("dtype", dtype)
      .setAttr("shape", tensor_shape_proto)
      .build()
  return Tensor(p, 0)
}

fun TF.placeholder(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Placeholder"): Tensor {
  val p = g.nodeBuilder("Placeholder", ctx.getUniqueFullName(name))
      .setAttrType("dtype", dtype)
      .setAttr("shape", shape)
      .build()
  return Tensor(p, 0)
}

fun TF.zerosLike(x: Tensor, name: String = "ZerosLike") =
    unaryOp("ZerosLike", x, name)

fun TF.onesLike(x: Tensor, name: String = "OnesLike") =
    unaryOp("OnesLike", x, name)

fun TF.zeros(shape: Tensor, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  TODO()
}

fun TF.zeros(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  subscope(name) {
    val zero = when (dtype) {
      DT_STRING -> ""
      else -> 0
    }
    return if (shape.numElements() < 1000)
      const(shape, dtype, zero, borrowParentName())
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, zero), borrowParentName())
    }
  }
}

fun TF.ones(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  subscope(name) {
    return if (shape.numElements() < 1000)
      const(shape, dtype, 1, borrowParentName())
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, 1), borrowParentName())
    }
  }
}

fun TF.fill(dims: Tensor, value: Tensor, name: String = "Fill") =
    binaryOp("Fill", dims, value, name)

fun TF.reshape(tensor: Tensor, shape: Tensor, name: String = "Reshape") =
    binaryOp("Reshape", tensor, shape, name)

fun TF.slice(input: Tensor, begin: Tensor, size: Tensor, name: String = "Slice")
    : Tensor {
  val v = g.nodeBuilder("Slice", ctx.getUniqueFullName(name))
      .addInput(input)
      .addInput(begin)
      .addInput(size)
      .build()
  return Tensor(v, 0)
}

fun TF.oneHot(indices: Tensor, depth: Tensor,
              on_value: Tensor = tf.const(1f),
              off_value: Tensor = tf.const(0f),
              name: String = "OneHot")
    : Tensor {
  val v = g.nodeBuilder("OneHot", ctx.getUniqueFullName(name))
      .addInput(indices)
      .addInput(depth)
      .addInput(on_value)
      .addInput(off_value)
      .build()
  return Tensor(v, 0)
}

/**
 * Returns the shape of a tensor.
 *
 *
 */
fun TF.shape(input: Tensor, name: String = "Shape", optimize: Boolean = true): Tensor {
  //TODO SparseTensor
  val out_type = DT_INT32
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.asIntArray(), name)
  val op = g.nodeBuilder("Shape", ctx.getUniqueFullName(name))
      .addInput(input)
      .setAttrType("out_type", out_type)
      .build()
  return Tensor(op, 0)
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
operator fun Tensor.get(vararg slice_spec: Int): Tensor {
  val begin = mutableListOf<Int>()
  val end = mutableListOf<Int>()
  val strides = mutableListOf<Int>()
  
  var shrink_axis_mask = 0
  var new_axis_mask = 0
  var begin_mask = 0
  var end_mask = 0
  var ellipsis_mask = 0
  for ((index, s) in slice_spec.withIndex()) {
    //TODO other class
    begin += s
    end += s + 1
    strides += 1
    shrink_axis_mask = shrink_axis_mask or (1 shl index)
  }
  
  tf.subscope("strided_slice") {
    val packed_begin = tf.stack(begin)
    val packed_end = tf.stack(end)
    val packed_strides = tf.stack(strides)
    return tf.strideSlice(this@get, packed_begin, packed_end, packed_strides,
                          StridedSliceAttrs(
                              begin_mask_ = begin_mask,
                              end_mask_ = end_mask,
                              shrink_axis_mask_ = shrink_axis_mask,
                              new_axis_mask_ = new_axis_mask,
                              ellipsis_mask_ = ellipsis_mask), name = parentName)
  }
}

class StridedSliceAttrs(var begin_mask_: Int = 0,
                        var end_mask_: Int = 0,
                        var ellipsis_mask_: Int = 0,
                        var new_axis_mask_: Int = 0,
                        var shrink_axis_mask_: Int = 0)

fun TF.strideSlice(input: Tensor, begin: Tensor, end: Tensor, strides: Tensor,
                   attrs: StridedSliceAttrs = StridedSliceAttrs(),
                   name: String = "StridedSlice"): Tensor {
  val op = g.nodeBuilder("StridedSlice", ctx.getUniqueFullName(name))
      .addInput(input)
      .addInput(begin)
      .addInput(end)
      .addInput(strides)
      .setAttr("begin_mask", attrs.begin_mask_)
      .setAttr("end_mask", attrs.end_mask_)
      .setAttr("ellipsis_mask", attrs.ellipsis_mask_)
      .setAttr("new_axis_mask", attrs.new_axis_mask_)
      .setAttr("shrink_axis_mask", attrs.shrink_axis_mask_)
      .build()
  return Tensor(op, 0)
}

fun TF.gather(params: Tensor, indices: Tensor, axis: Int = 0, name: String = "GatherV2"): Tensor {
  if (axis == 0) {
  
  }
  //TODO detect resource variables
  val op = g.nodeBuilder("GatherV2", ctx.getUniqueFullName(name))
      .addInput(params)
      .addInput(indices)
      .addInput(const(axis))
      .build()
  return Tensor(op, 0)
}

fun TF.rank(input: Tensor, name: String = "Rank", optimize: Boolean = true): Tensor {
  //TODO SparseTensor
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.rank(), name)
  return unaryOp("Rank", input, name)
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
 * @param values A list of `Tensor` objects with the same shape and type.
 * @param axis An `int`. The axis to stack along. Defaults to the first dimension.
 * Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
 * @param name
 * @return A stacked `Tensor` with the same type as `values`.
 */
fun TF.stack(values: Array<Tensor>, axis: Int = 0, name: String = "stack"): Tensor {
  if (axis == 0) {
    return pack(values, axis, name)
  }
  TODO()
}

fun TF.stack(values: List<Int>, axis: Int = 0, name: String = "stack"): Tensor {
  if (axis == 0) {
    return const(values.toIntArray(), name)
  }
  TODO()
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
fun TF.where(condition: Tensor, x: Tensor, y: Tensor) = ternaryOp("Select",condition, x, y, "Select")

fun TF.where(condition: Tensor, name: String = "Where") = unaryOp("Where", condition, name)

/**Tensor conversion function that automatically packs arguments.*/
fun TF.autopack(v: Array<Tensor>, name: String = "packed"): Tensor {
  TODO()
}


fun TF.pack(value: Array<Tensor>, axis: Int = 0, name: String = "Pack"): Tensor {
  val op = g.nodeBuilder("Pack", ctx.getUniqueFullName(name))
      .addInputList(value)
      .setAttr("axis", axis)
      .build()
  return Tensor(op, 0)
}

fun TF.stop_gradient(input: Tensor, name: String = "StopGradient") = unaryOp("StopGradient", input, name)