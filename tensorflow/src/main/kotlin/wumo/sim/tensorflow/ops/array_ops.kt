package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape

object array_ops {
  interface API {
    fun <T : OutputLike> identity(data: T, name: String): Output {
      return when (data) {
        is Output -> {
          if (data.dataType.isRefType)
            tf._refIdentity(data, name)
          else
            tf._identity(data, name)
        }
        is IndexedSlices -> TODO()
        is SparseOutput -> TODO()
        else -> TODO()
      }
    }
    
    fun oneHot(indices: Output, depth: Output, on_value: Output = tf.const(1f, "on_value"),
               off_value: Output = tf.const(0f, "off_value"), axis: Long = -1L, name: String = "OneHot") =
        tf._oneHot(indices, depth, on_value, off_value, axis, name)
    
    fun placeholder(shape: Shape = Shape(),
                    dtype: DataType<*> = FLOAT, name: String = "Placeholder"): Output =
        tf._placeholder(dtype, shape, name)
    
    fun zerosLike(x: Output, dtype: DataType<*>? = null, optimize: Boolean = true, name: String = "ZerosLike") =
        when {
          optimize && x.shape.isFullyDefined && x.dataType != VARIANT ->
            zeros(x.shape, dtype = dtype ?: x.dataType, name = name)
          dtype != null && dtype != x.dataType && dtype != VARIANT ->
            zeros(shape(x, optimize = optimize), dtype = dtype, name = name)
          else -> tf._zerosLike(x, name)
        }
    
    fun zeros(shape: Output, dtype: DataType<*> = FLOAT, name: String = "Ones"): Output {
      TODO()
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
            val shape = tf._reshape(tf.const(shape.asLongArray()), tf.const(-1))
            tf._fill(shape, tf.const(dtype, zero), tf.currentNameScope)
          }
        }
    
    fun ones(shape: Shape, dtype: DataType<*> = FLOAT, name: String = "Ones"): Output =
        tf.nameScope(name) {
          if (shape.numElements() < 1000)
            tf.const(shape, dtype, 1, tf.currentNameScope)
          else {
            val shape = tf._reshape(tf.const(shape.asLongArray()), tf.const(-1))
            tf._fill(shape, tf.const(dtype, 1), tf.currentNameScope)
          }
        }
    
    /**
     * Returns the shape of a tensor.
     *
     *
     */
    fun shape(input: Output, out_type: DataType<*> = INT32, name: String = "Shape", optimize: Boolean = true): Output {
      //TODO SparseOutput
      val input_shape = input.shape
      if (optimize && input_shape.isFullyDefined)
        return tf.const(input_shape.asIntArray()!!, name)
      return tf._shape(input, out_type, name)
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
      return tf.nameScope("strided_slice") {
        val packed_begin = stack(begin)
        val packed_end = stack(end)
        val packed_strides = stack(strides)
        tf._stridedSlice(this@get, packed_begin, packed_end, packed_strides,
                         begin_mask,
                         end_mask,
                         shrink_axis_mask,
                         new_axis_mask,
                         ellipsis_mask,
                         tf.currentNameScope)
      }
    }
    
    class StridedSliceAttrs(var begin_mask_: Int = 0,
                            var end_mask_: Int = 0,
                            var ellipsis_mask_: Int = 0,
                            var new_axis_mask_: Int = 0,
                            var shrink_axis_mask_: Int = 0)
    
    fun gather(params: Output, indices: Output, axis: Int = 0, name: String = "GatherV2"): Output {
      if (axis == 0) {
      }
      //TODO detect resource variables
      return tf._gatherV2(params, indices, tf.const(axis), name)
    }
    
    fun rank(input: Output, name: String = "Rank", optimize: Boolean = true): Output {
      //TODO SparseOutput
      val input_shape = input.shape
      if (optimize && input_shape.isFullyDefined)
        return tf.const(input_shape.rank, name)
      return tf._rank(input, name)
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
    fun stack(values: List<Output>, axis: Long = 0L, name: String = "stack"): Output {
      if (axis == 0L) {
        return tf._pack(values, axis, name)
      }
      TODO()
    }
    
    fun stack(values: List<Int>, axis: Int = 0, name: String = "stack"): Output {
      if (axis == 0) {
        return tf.const(values.toIntArray(), name)
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
    fun where(condition: Output, x: Output, y: Output, name: String = "Where") =
        tf._select(condition, x, y, name)
    
    fun where(condition: Output, name: String = "Where") = tf._where(condition, name)
    /**Output conversion function that automatically packs arguments.*/
    fun autopack(v: Array<Output>, name: String = "packed"): Output {
      TODO()
    }
  }
}
