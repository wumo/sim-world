package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.ops.gen.gen_math_ops
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.arange

operator fun <T : OutputConvertible, R : OutputConvertible> T.rem(b: R) =
    tf.floorMod(this.toOutput(), b.toOutput())

infix fun <T : OutputConvertible, R : OutputConvertible> T.and(b: R) =
    tf.logicalAnd(this.toOutput(), b.toOutput())

infix fun <T : OutputConvertible, R : OutputConvertible> T.or(b: R) =
    tf.logicalOr(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible, R : OutputConvertible> T.not() =
    tf.logicalNot(this.toOutput())

operator fun <T : OutputConvertible> T.plus(b: Any) =
    tf.nameScope("add") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.baseDataType, b, name = "y")
      tf.add(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible> Any.plus(b: T) =
    tf.nameScope("add") {
      val a = b.toOutput()
      val y = tf.const(a.dataType.baseDataType, this, name = "y")
      tf.add(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.plus(b: R) =
    tf.add(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> T.div(b: Any) =
    tf.nameScope("div") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.baseDataType, b, name = "y")
      tf.div(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible> Any.div(b: T) =
    tf.nameScope("div") {
      val a = b.toOutput()
      val y = tf.const(a.dataType.baseDataType, this, name = "y")
      tf.div(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.div(b: R) =
    tf.div(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> T.minus(b: Any) =
    tf.nameScope("sub") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.baseDataType, b, name = "y")
      tf.sub(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible> Any.minus(b: T) =
    tf.nameScope("sub") {
      val a = b.toOutput()
      val y = tf.const(a.dataType.baseDataType, this, name = "y")
      tf.sub(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.minus(b: R) =
    tf.sub(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> Any.times(b: T) =
    tf.nameScope("mul") {
      val a = b.toOutput()
      val y = tf.const(a.dataType.baseDataType, this, name = "y")
      tf.mul(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible> T.times(b: Any) =
    tf.nameScope("mul") {
      val a = this.toOutput()
      val y = tf.const(a.dataType.baseDataType, b, name = "y")
      tf.mul(a, y, name = tf.currentNameScope)
    }

operator fun <T : OutputConvertible, R : OutputConvertible> T.times(b: R) =
    tf.mul(this.toOutput(), b.toOutput())

operator fun <T : OutputConvertible> T.unaryMinus() = tf.neg(this.toOutput())

object math_ops {
  private val dtype_hierarchy: Map<DataType<*>, Int> = mapOf(INT32 to 0,
                                                             INT64 to 1,
                                                             FLOAT to 2,
                                                             DOUBLE to 3)
  
  interface API : gen_math_ops {
    fun accumulateN(inputs: List<Output>, shape: Shape? = null, name: String = "AccumulateNV2"): Output {
      val dataType = inputs[0].dataType
      if (inputs.any { it.dataType != dataType })
        throw InvalidArgumentException("All input tensors must have the same data type.")
      val inferredShape = shape ?: Shape()
      if (inputs.any { !it.shape.isCompatibleWith(inferredShape) })
        throw InvalidArgumentException("All input tensors must have the same shape.")
      return when {
        inputs.size == 1 && name.isEmpty() -> inputs[0]
        inputs.size == 1 -> tf.identity(inputs[0], name)
        else -> super.accumulateNV2(inputs, inferredShape, name)
      }
    }
    
    fun argmax(a: Output, axis: Int = 0, output_type: DataType<*> = INT64, name: String = "ArgMax") =
        tf.nameScope(name) {
          val dimension = tf.const(axis, "dimension")
          super.argMax(a, dimension, output_type, tf.currentNameScope)
        }
    
    fun argmin(a: Output, axis: Int = 0, output_type: DataType<*> = INT64, name: String = "ArgMin") =
        tf.nameScope(name) {
          val dimension = tf.const(axis, "dimension")
          super.argMin(a, dimension, output_type, tf.currentNameScope)
        }
    
    fun cast(x: Output, dstT: DataType<*>, name: String = "Cast"): Output = run {
      val x = (x as? Variable)?.value ?: x
      if (x.dataType == dstT) x
      else super.cast(x, dstT, false, name)
    }
    
    override fun conj(x: Output, name: String) =
        when {
          x.dataType.isComplex -> super.conj(x, name)
          x.dataType.isNumeric -> x
          else -> throw IllegalArgumentException("'conjugate' can only take numeric tensors as input.")
        }
    
    fun greaterEqual(a: Output, b: Any, name: String = "GreaterEqual") =
        tf.nameScope(name) {
          val y = tf.const(a.dataType.baseDataType, b, name = "y")
          super.greaterEqual(a, y, tf.currentNameScope)
        }
    
    fun reducedShape(inputShape: Output, axes: Output): Output {
      val inputShape = tf.cast(inputShape, INT32)
      var axes = tf.cast(axes, INT32)
      
      val inputRank = tf.size(inputShape)
      axes = (axes + inputRank) % inputRank
      val axesShape = tf.shape(axes)
      return tf.dynamicStitch(
          listOf(tf.range(tf.const(0), inputRank),
                 axes),
          listOf(inputShape,
                 tf.fill(axesShape, tf.const(1))))
    }
    
    fun reductionDims(x: Output, axis: Output?): Output {
      if (axis != null) return axis
      //Fast path: avoid creating Rank and Range ops if ndims is known.
      return tf.const(arange(x.shape.rank), name = "reduction_indices")
      //TODO SparseOutput
      // Otherwise, we rely on Range and Rank to do the right thing at run-time.
      return range(tf.const(0), tf.rank(x))
    }
    
    fun mean(input: Output, axis: LongArray? = null, keep_dims: Boolean = false, name: String = "mean") =
        tf.nameScope(name) {
          val reduction_indices =
              reductionDims(input,
                            if (axis != null) tf.const(axis, "reduction_indices")
                            else null)
          super.mean(input, reduction_indices, keep_dims, name)
        }
    
    fun matMul(a: Output, b: Output, transposeA: Boolean = false, transposeB: Boolean = false,
               conjugateA: Boolean = false,
               conjugateB: Boolean = false,
               aIsSparse: Boolean = false,
               bIsSparse: Boolean = false,
               name: String = "MatMul"): Output =
        tf.nameScope(name, setOf(a.op, b.op)) {
          val (cA, cB) = castArgs(a, b)
          val sparseMatMulDataTypes = setOf(BFLOAT16, FLOAT)
          if (!aIsSparse && !bIsSparse
              && (cA.rank == -1 || cA.rank > 2)
              && (cB.rank == -1 || cB.rank > 2)) {
            // "BatchMatMul" does not support transpose, so we conjugate the matrix and use adjoint instead.
            // The "conj" op is a no-op for real matrices.
            val (x, adjointX) = transposeConjugateToAdjoint(cA, transposeA, conjugateA)
            val (y, adjointY) = transposeConjugateToAdjoint(cB, transposeB, conjugateB)
            super.batchMatMul(x, y, adjointX, adjointY, tf.currentNameScope)
          } else if (cA.dataType == BFLOAT16 || cB.dataType == BFLOAT16 || // "MatMul" does not currently support this type.
              ((aIsSparse || bIsSparse) &&
                  sparseMatMulDataTypes.contains(cA.dataType) &&
                  sparseMatMulDataTypes.contains(cB.dataType))) {
            val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
            val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
            super.sparseMatMul(x, y, transposeX, transposeY, aIsSparse, bIsSparse, tf.currentNameScope)
          } else {
            val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
            val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
            super.matMul(x, y, transposeX, transposeY, tf.currentNameScope)
          }
        }
    
    fun castArgs(a: Output, b: Output): Pair<Output, Output> {
      val dataType = DataType.mostPrecise(a.dataType, b.dataType)
      val cA = tf.cast(a, dataType)
      val cB = tf.cast(b, dataType)
      return Pair(cA, cB)
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
      super._range(start, limit, delta, name)
    }
    
    fun realDiv(a: Output, b: Any, name: String = "RealDiv") =
        tf.nameScope("truediv") {
          val y = tf.const(a.dataType.baseDataType, b, name = "y")
          super.realDiv(a, y, name = tf.currentNameScope)
        }
    
    fun scalarMul(scalar: Output, x: Output, name: String = "scalarMul"): Output {
      return tf.mul(scalar, x)
    }
    
    fun sum(input: Output, axis: Int? = null, keep_dims: Boolean = false, name: String = "sum") =
        tf.nameScope(name) {
          val reduction_indices =
              reductionDims(input,
                            if (axis != null) tf.const(axis, "reduction_indices")
                            else null)
          super._sum(input, reduction_indices, keep_dims, tf.currentNameScope)
        }
    
    fun sum(input: Output, axis: Output? = null, keep_dims: Boolean = false, name: String = "sum") =
        super._sum(input, reductionDims(input, axis), keep_dims, name)
    
    fun prod(input: Output, axis: Output? = null, keep_dims: Boolean = false, name: String = "Prod") =
        super._prod(input, reductionDims(input, axis), keep_dims, name)
    
    fun transpose(input: Output, perm: Output? = null, conjugate: Boolean = false,
                  name: String = "Transpose"): Output {
      val transposeFn: (Output, Output, String) -> Output = if (conjugate && input.dataType.isComplex)
        tf::conjugateTranspose
      else
        tf::transpose
      return if (perm == null) {
        val rank = tf.rank(input)
        val perm = (rank - 1) - tf.range(tf.const(0), rank, tf.const(1))
        val transposed = transposeFn(input, perm, name)
        val inputShape = transposed.op.inputs[0].shape
        if (inputShape.rank != -1)
          transposed.setShape(Shape(inputShape.asIntArray()!!.reversedArray()))
        transposed
      } else
        transposeFn(input, perm, name)
    }
    
    fun tensordot(input: Output, kernel: Output, const: Output): Output {
      TODO()
    }
  }
  
  private fun transposeConjugateToAdjoint(
      tensor: Output, transpose: Boolean, conj: Boolean): Pair<Output, Boolean> =
      if (transpose) {
        if (conj)
          tensor to true
        else
          tf.conj(tensor) to true
      } else {
        if (conj)
          tf.conj(tensor) to false
        else
          tensor to false
      }
  
  private fun transposeConjugateToTranspose(
      tensor: Output, transpose: Boolean, conj: Boolean): Pair<Output, Boolean> =
      if (transpose) {
        if (conj)
          tf.conj(tensor) to true
        else
          tensor to true
      } else {
        if (conj)
          tf.conj(tensor) to false
        else
          tensor to false
      }
}