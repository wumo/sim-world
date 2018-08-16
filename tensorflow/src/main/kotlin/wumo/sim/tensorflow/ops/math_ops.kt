package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.a
import wumo.sim.util.arange

operator fun Output.plus(b: Any) =
    tf.nameScope("add") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf._add(this, y, name = tf.currentNameScope)
    }

operator fun Output.plus(b: Output) = tf._add(this, b)

operator fun Output.div(b: Any) =
    tf.nameScope("div") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf._div(this, y, name = tf.currentNameScope)
    }

operator fun Output.div(b: Output) = tf._div(this, b)

operator fun Output.minus(b: Any) =
    tf.nameScope("sub") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf._sub(this, y, name = tf.currentNameScope)
    }

operator fun Output.minus(b: Output) = tf._sub(this, b)

operator fun Output.times(b: Any) =
    tf.nameScope("mul") {
      val y = tf.const(this.dtype.base_dtype, b, name = "y")
      tf._mul(this, y, name = tf.currentNameScope)
    }

operator fun Output.times(b: Output) = tf._mul(this, b)

operator fun Output.unaryMinus() = tf._neg(this)

object math_ops {
  private val dtype_hierarchy: Map<DataType<*>, Int> = mapOf(INT32 to 0,
                                                             INT64 to 1,
                                                             FLOAT to 2,
                                                             DOUBLE to 3)
  
  interface API {
    fun argmax(a: Output, axis: Int = 0, output_type: DataType<*> = INT64, name: String = "ArgMax") =
        tf.nameScope(name) {
          val dimension = tf.const(axis, "dimension")
          tf._argMax(a, dimension, output_type, tf.currentNameScope)
        }
    
    fun argmin(a: Output, axis: Int = 0, output_type: DataType<*> = INT64, name: String = "ArgMin") =
        tf.nameScope(name) {
          val dimension = tf.const(axis, "dimension")
          tf._argMin(a, dimension, output_type, tf.currentNameScope)
        }
    
    fun cast(x: Output, dstT: DataType<*>, name: String = "Cast"): Output = run {
      val x = (x as? Variable)?.value ?: x
      if (x.dtype == dstT) x
      else tf._cast(x, dstT, name = name)
    }
    
    fun conj(x: Output, name: String = "Conj") =
        if (x.dtype == COMPLEX64 || x.dtype == COMPLEX128)
          tf._conj(x, name)
        else
          x
    
    fun greaterEqual(a: Output, b: Any, name: String = "GreaterEqual") =
        tf.nameScope(name) {
          val y = tf.const(a.dtype.base_dtype, b, name = "y")
          tf._greaterEqual(a, y, tf.currentNameScope)
        }
    
    fun reductionDims(x: Output, axis: Output?): Output {
      if (axis != null) return axis
      //Fast path: avoid creating Rank and Range ops if ndims is known.
      return tf.const(arange(x.shape.rank), name = "reduction_indices")
      //TODO SparseOutput
      // Otherwise, we rely on Range and Rank to do the right thing at run-time.
      return range(tf.const(0), tf._rank(x))
    }
    
    fun mean(input: Output, axis: LongArray? = null, keep_dims: Boolean = false, name: String = "mean") =
        tf.nameScope(name) {
          val reduction_indices =
              reductionDims(input,
                            if (axis != null) tf.const(axis, "reduction_indices")
                            else null)
          tf._mean(input, reduction_indices, keep_dims, name)
        }
    
    /**
    Creates a sequence of numbers.
    
    Creates a sequence of numbers that begins at `start` and extends by
    increments of `delta` up to but not including `limit`.
    
    The dtype of the resulting tensor is inferred from the inputs unless
    it is provided explicitly.
    
    Like the Python builtin `range`, `start` defaults to 0, so that
    `range(n) = range(0, n)`.
    
    For example:
    
    ```python
    start = 3
    limit = 18
    delta = 3
    range(start, limit, delta)  # [3, 6, 9, 12, 15]
    
    start = 3
    limit = 1
    delta = -0.5
    range(start, limit, delta)  # [3, 2.5, 2, 1.5]
    
    limit = 5
    range(limit)  # [0, 1, 2, 3, 4]
    ```
     * @param start: A 0-D `Output` (scalar). Acts as first entry in the range if
     * `limit` is not None; otherwise, acts as range limit and first entry
     * defaults to 0.
     * @param limit: A 0-D `Output` (scalar). Upper limit of sequence,
     * exclusive. If None, defaults to the value of `start` while the first
     * entry of the range defaults to 0.
     * @param delta: A 0-D `Output` (scalar). Number that increments
     * `start`. Defaults to 1.
     * @param name: A name for the operation. Defaults to "range".
     * @return An 1-D `Output` of type `dtype`.
     */
    fun range(start: Output, limit: Output, delta: Output = tf.const(1), name: String = "Range") = run {
      val dtypes = a(start.dtype, limit.dtype, delta.dtype)
      val inferred_dtype = dtypes.maxBy { dtype_hierarchy[it]!! }!!
      val start = cast(start, inferred_dtype)
      val limit = cast(limit, inferred_dtype)
      val delta = cast(delta, inferred_dtype)
      tf._range(start, limit, delta, name)
    }
    
    fun realDiv(a: Output, b: Any, name: String = "RealDiv") =
        tf.nameScope("truediv") {
          val y = tf.const(a.dtype.base_dtype, b, name = "y")
          tf._realDiv(a, y, name = tf.currentNameScope)
        }
    
    fun sum(input: Output, axis: Int? = null, keep_dims: Boolean = false, name: String = "sum") =
        tf.nameScope(name) {
          val reduction_indices =
              reductionDims(input,
                            if (axis != null) tf.const(axis, "reduction_indices")
                            else null)
          tf._sum(input, reduction_indices, keep_dims, tf.currentNameScope)
        }
    
    fun sum(input: Output, axis: Output? = null, keep_dims: Boolean = false, name: String = "sum") =
        tf._sum(input, reductionDims(input, axis), keep_dims, name)
    
    fun tensordot(input: Output, kernel: Output, const: Output): Output {
      TODO()
    }
  }
}