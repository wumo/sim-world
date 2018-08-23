package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.arange

operator fun <T : OutputConvertible> T.plus(b: Any) =
    tf.nameScope("add") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.base_dtype, b, name = "y")
      tf._add(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.plus(b: R) =
    tf._add(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> T.div(b: Any) =
    tf.nameScope("div") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.base_dtype, b, name = "y")
      tf._div(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.div(b: R) =
    tf._div(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> T.minus(b: Any) =
    tf.nameScope("sub") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.base_dtype, b, name = "y")
      tf._sub(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.minus(b: R) =
    tf._sub(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> T.times(b: Any) =
    tf.nameScope("mul") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.base_dtype, b, name = "y")
      tf._mul(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.times(b: R) =
    tf._mul(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> T.unaryMinus() = tf._neg(this.toOutput())

object math_ops {
  private val dtype_hierarchy: Map<DataType<*>, Int> = mapOf(INT32 to 0,
                                                             INT64 to 1,
                                                             FLOAT to 2,
                                                             DOUBLE to 3)
  
  interface API {
    fun accumulateNV2(inputs: List<Output>, shape: Shape? = null, name: String = "AccumulateNV2"): Output {
      val dataType = inputs[0].dataType
      if (inputs.any { it.dataType != dataType })
        throw InvalidArgumentException("All input tensors must have the same data type.")
      val inferredShape = shape ?: Shape()
      if (inputs.any { !it.shape.isCompatibleWith(inferredShape) })
        throw InvalidArgumentException("All input tensors must have the same shape.")
      return when {
        inputs.size == 1 && name.isEmpty() -> inputs[0]
        inputs.size == 1 -> tf.identity(inputs[0], name)
        else -> tf._accumulateNV2(inputs, inferredShape, name)
      }
    }
    
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
      if (x.dataType == dstT) x
      else tf._cast(x, dstT, name = name)
    }
    
    fun conj(x: Output, name: String = "Conj") =
        if (x.dataType == COMPLEX64 || x.dataType == COMPLEX128)
          tf._conj(x, name)
        else
          x
    
    fun greaterEqual(a: Output, b: Any, name: String = "GreaterEqual") =
        tf.nameScope(name) {
          val y = tf.const(a.dataType.base_dtype, b, name = "y")
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
    
    The dataType of the resulting tensor is inferred from the inputs unless
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
     * @return An 1-D `Output` of type `dataType`.
     */
    fun range(start: Output, limit: Output, delta: Output = tf.const(1), name: String = "Range") = run {
      val dtypes = a(start.dataType, limit.dataType, delta.dataType)
      val inferred_dtype = dtypes.maxBy { dtype_hierarchy[it]!! }!!
      val start = cast(start, inferred_dtype)
      val limit = cast(limit, inferred_dtype)
      val delta = cast(delta, inferred_dtype)
      tf._range(start, limit, delta, name)
    }
    
    fun realDiv(a: Output, b: Any, name: String = "RealDiv") =
        tf.nameScope("truediv") {
          val y = tf.const(a.dataType.base_dtype, b, name = "y")
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