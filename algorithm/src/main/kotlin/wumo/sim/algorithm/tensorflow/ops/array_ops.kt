package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.util.Dimension

fun TF.batchToSpace(input: Tensor, crops: Tensor, block_size: Long, name: String = "BatchToSpace") =
    naryOp("BatchToSpace", input.value(), crops.value(), name = name) {
      attr("block_size", block_size)
    }

fun TF.batchToSpaceND(input: Tensor, block_shape: Tensor, crops: Tensor, name: String = "BatchToSpaceND") =
    ternaryOp("BatchToSpaceND", input.value(), block_shape.value(), crops, name)

fun TF.checkNumerics(tensor: Tensor, message: String, name: String = "CheckNumerics") =
    naryOp("CheckNumerics", tensor.value(), name = name) {
      attr("messge", message)
    }

fun TF.concat(values: Collection<Tensor>, axis: Tensor, name: String = "ConcatV2") =
    naryOp("ConcatV2", name = name) {
      addInput(values.map { it.value() })
      addInput(axis)
    }

fun TF.concat(values: Array<Tensor>, axis: Tensor, name: String = "ConcatV2") =
    naryOp("ConcatV2", name = name) {
      addInput(values.map { it.value() })
      addInput(axis)
    }

fun TF.depthToSpace(input: Tensor, block_size: Long, data_format: String = "NHWC", name: String = "DepthToSpace") =
    naryOp("DepthToSpace", input.value(), name = name) {
      attr("block_size", block_size)
      attr("data_format", data_format)
    }

fun TF.diag(diagonal: Tensor, name: String = "Diag") = unaryOp("Diag", diagonal.value(), name)
fun TF.diagPart(input: Tensor, name: String = "DiagPart") = unaryOp("DiagPart", input.value(), name)
fun TF.gatherNd(params: Tensor, indices: Tensor, name: String = "GatherNd") =
    binaryOp("GatherNd", params.value(), indices.value(), name)

fun TF.gatherV2(params: Tensor, indices: Tensor, axis: Tensor, name: String = "GatherV2") =
    ternaryOp("GatherV2", params.value(), indices.value(), axis.value(), name)

fun TF.identity(input: Tensor, name: String = "Identity") =
    unaryOp("Identity", input, name)

fun TF.invertPermutation(x: Tensor, name: String = "InvertPermutation") = unaryOp("InvertPermutation", x.value(), name)
fun TF.matrixBandPart(input: Tensor, num_lower: Tensor, num_upper: Tensor, name: String = "MatrixBandPart") =
    ternaryOp("MatrixBandPart", input.value(), num_lower.value(), num_upper.value(), name)

fun TF.matrixDiag(diagonal: Tensor, name: String = "MatrixDiag") = unaryOp("MatrixDiag", diagonal.value(), name)
fun TF.matrixDiagPart(input: Tensor, name: String = "MatrixDiagPart") = unaryOp("MatrixDiagPart", input.value(), name)
fun TF.matrixSetDiag(input: Tensor, diagonal: Tensor, name: String = "MatrixSetDiag") =
    binaryOp("MatrixSetDiag", input.value(), diagonal.value(), name)

fun TF.mirrorPad(input: Tensor, paddings: Tensor,
                 mode: String,
                 name: String = "MirrorPad") =
    naryOp("MirrorPad", input.value(), paddings.value(), name = name) {
      attr("mode", mode)
    }

fun TF.pad(input: Tensor, paddings: Tensor, name: String = "Pad") =
    binaryOp("Pad", input.value(), paddings.value(), name)

fun TF.padV2(input: Tensor, paddings: Tensor, constant_values: Tensor, name: String = "PadV2") =
    ternaryOp("PadV2", input.value(), paddings.value(), constant_values.value(), name)

fun TF.placeholder(dtype: Int = DT_FLOAT, name: String = "Placeholder") =
    naryOp("Placeholder", name = name) {
      val tensor_shape_proto = TensorShapeProto().apply { set_unknown_rank(true) }
      tensor_shape_proto.set_unknown_rank(true)
      attrType("dtype", dtype)
      attr("shape", tensor_shape_proto)
    }

fun TF.placeholder(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Placeholder") =
    naryOp("Placeholder", name = name) {
      val tensor_shape_proto = TensorShapeProto()
      tensor_shape_proto.set_unknown_rank(true)
      attrType("dtype", dtype)
      attr("shape", shape)
    }

fun TF.reverseSequence(input: Tensor, seq_lengths: Tensor,
                       seq_dim: Long, batch_dim: Long = 0,
                       name: String = "ReverseSequence") =
    naryOp("ReverseSequence", input.value(), seq_lengths.value(), name = name) {
      attr("seq_dim", seq_dim)
      attr("batch_dim", batch_dim)
    }

fun TF.reverse(tensor: Tensor, axis: Tensor, name: String = "Reverse") =
    binaryOp("Reverse", tensor.value(), axis.value(), name = name)

fun TF.spaceToBatch(input: Tensor, paddings: Tensor, block_size: Long, name: String = "SpaceToBatch") =
    naryOp("SpaceToBatch", input.value(), paddings.value(), name = name) {
      attr("block_size", block_size)
    }

fun TF.spaceToBatchND(input: Tensor, block_shape: Tensor, paddings: Tensor, name: String = "SpaceToBatchND") =
    ternaryOp("SpaceToBatchND", input.value(), block_shape.value(), paddings, name)

fun TF.spaceToDepth(input: Tensor, block_size: Long, data_format: String = "NHWC", name: String = "SpaceToDepth") =
    naryOp("SpaceToDepth", input.value(), name = name) {
      attr("block_size", block_size)
      attr("data_format", data_format)
    }

fun TF.stridedSliceGrad(shape: Tensor, begin: Tensor, end: Tensor, strides: Tensor, dy: Tensor,
                        begin_mask: Long = 0L, end_mask: Long = 0L, ellipsis_mask: Long = 0L,
                        new_axis_mask: Long = 0L, shrink_axis_mask: Long = 0L,
                        name: String = "StridedSliceGrad") =
    naryOp("StridedSliceGrad", shape.value(), begin.value(), end.value(), strides.value(), dy.value(), name = name) {
      attr("begin_mask", begin_mask)
      attr("end_mask", end_mask)
      attr("ellipsis_mask", ellipsis_mask)
      attr("new_axis_mask", new_axis_mask)
      attr("shrink_axis_mask", shrink_axis_mask)
    }

fun TF.scatterNd(indices: Tensor, updates: Tensor, shape: Tensor, name: String = "ScatterNd") =
    ternaryOp("ScatterNd", indices.value(), updates.value(), shape.value(), name)

fun TF.squeeze(input: Tensor, axis: IntArray = IntArray(0), name: String = "Squeeze") =
    naryOp("Squeeze", input.value(), name = name) {
      attr("squeeze_dims", axis)
    }

fun TF.listDiff(x: Tensor, y: Tensor, out_idx: Int = DT_INT32, name: String = "ListDiff") =
    naryOps("ListDiff", x.value(), y.value(), name = name) {
      attrType("out_idx", out_idx)
    }

fun TF.zerosLike(x: Tensor, dtype: Int = DT_INVALID, optimize: Boolean = true, name: String = "ZerosLike") =
    when {
      optimize && x.shape.is_fully_defined && x.dtype != DT_VARIANT ->
        zeros(x.shape, dtype = dtype.orUse(x.dtype), name = name)
      dtype != DT_INVALID && dtype != x.dtype && dtype != DT_VARIANT ->
        zeros(shape(x, optimize = optimize), dtype = dtype, name = name)
      else -> _zerosLike(x, name)
    }

fun TF._zerosLike(x: Tensor, name: String = "ZerosLike") =
    unaryOp("ZerosLike", x.value(), name)

fun TF._onesLike(x: Tensor, name: String = "OnesLike") =
    unaryOp("OnesLike", x.value(), name)

fun TF.zeros(shape: Tensor, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  TODO()
}

fun TF.zeros(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
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

fun TF.ones(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  name_scope(name) {
    return if (shape.numElements() < 1000)
      const(shape, dtype, 1, ctxNs.scopeName)
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, 1), ctxNs.scopeName)
    }
  }
}

fun TF.fill(dims: Tensor, value: Tensor, name: String = "Fill") =
    binaryOp("Fill", dims.value(), value.value(), name)

fun TF.reshape(tensor: Tensor, shape: Tensor, name: String = "Reshape") =
    binaryOp("Reshape", tensor.value(), shape.value(), name)

fun TF.slice(input: Tensor, begin: Tensor, size: Tensor, name: String = "Slice") =
    naryOp("Slice", input.value(), begin.value(), size, name = name)

fun TF.oneHot(indices: Tensor, depth: Tensor,
              on_value: Tensor = tf.const(1f),
              off_value: Tensor = tf.const(0f),
              name: String = "OneHot") =
    naryOp("OneHot", indices.value(), depth.value(), on_value.value(), off_value.value(), name = name)

fun TF.size(input: Tensor, out_type: Int = DT_INT32, name: String = "Size") =
    naryOp("Size", input.value(), name = name) {
      attrType("out_type", out_type)
    }

/**
 * Returns the shape of a tensor.
 *
 *
 */
fun TF.shape(input: Tensor, out_type: Int = DT_INT32, name: String = "Shape", optimize: Boolean = true): Tensor {
  //TODO SparseTensor
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.asIntArray(), name)
  return naryOp("Shape", input.value(), name = name) {
    attrType("out_type", out_type)
  }
}

fun TF.tile(input: Tensor, multiples: Tensor, name: String = "Tile") =
    binaryOp("Tile", input.value(), multiples.value(), name)

fun TF.transpose(x: Tensor, perm: Tensor, name: String = "Transpose") =
    binaryOp("Transpose", x.value(), perm.value(), name)

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
  tf.name_scope("strided_slice") {
    val packed_begin = tf.stack(begin)
    val packed_end = tf.stack(end)
    val packed_strides = tf.stack(strides)
    return tf.strideSlice(this@get, packed_begin, packed_end, packed_strides,
                          StridedSliceAttrs(
                              begin_mask_ = begin_mask,
                              end_mask_ = end_mask,
                              shrink_axis_mask_ = shrink_axis_mask,
                              new_axis_mask_ = new_axis_mask,
                              ellipsis_mask_ = ellipsis_mask), name = tf.ctxNs.scopeName)
  }
}

class StridedSliceAttrs(var begin_mask_: Int = 0,
                        var end_mask_: Int = 0,
                        var ellipsis_mask_: Int = 0,
                        var new_axis_mask_: Int = 0,
                        var shrink_axis_mask_: Int = 0)

fun TF.strideSlice(input: Tensor, begin: Tensor, end: Tensor, strides: Tensor,
                   attrs: StridedSliceAttrs = StridedSliceAttrs(),
                   name: String = "StridedSlice") =
    naryOp("StridedSlice", input.value(), begin.value(), end.value(), strides.value(), name = name) {
      attr("begin_mask", attrs.begin_mask_)
      attr("end_mask", attrs.end_mask_)
      attr("ellipsis_mask", attrs.ellipsis_mask_)
      attr("new_axis_mask", attrs.new_axis_mask_)
      attr("shrink_axis_mask", attrs.shrink_axis_mask_)
    }

fun TF.gather(params: Tensor, indices: Tensor, axis: Int = 0, name: String = "GatherV2"): Tensor {
  if (axis == 0) {
  }
  //TODO detect resource variables
  return naryOp("GatherV2", params.value(), indices.value(), const(axis), name = name)
}

fun TF.rank(input: Tensor, name: String = "Rank", optimize: Boolean = true): Tensor {
  //TODO SparseTensor
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.rank(), name)
  return unaryOp("Rank", input.value(), name)
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

fun TF.stack(values: Collection<Tensor>, axis: Int = 0, name: String = "stack") =
    tf.pack(values, axis, name)

fun TF.unstack(value: Tensor, num: Long, axis: Long = 0L, name: String = "Unstack") =
    naryOps("Unpack", value.value(), name = name) {
      attr("num", num)
      attr("axis", axis)
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
fun TF.where(condition: Tensor, x: Tensor, y: Tensor, name: String = "Where") =
    ternaryOp("Select", condition.value(), x.value(), y.value(), name)

fun TF.where(condition: Tensor, name: String = "Where") = unaryOp("Where", condition.value(), name)
/**Tensor conversion function that automatically packs arguments.*/
fun TF.autopack(v: Array<Tensor>, name: String = "packed"): Tensor {
  TODO()
}

fun TF.pack(value: Collection<Tensor>, axis: Int = 0, name: String = "Pack") =
    naryOp("Pack", name = name) {
      addInput(value.map { it.value() })
      attr("axis", axis)
    }

fun TF.pack(value: Array<Tensor>, axis: Int = 0, name: String = "Pack") =
    naryOp("Pack", name = name) {
      addInput(value.map { it.value() })
      attr("axis", axis)
    }

fun TF.stop_gradient(input: Tensor, name: String = "StopGradient") = unaryOp("StopGradient", input.value(), name)