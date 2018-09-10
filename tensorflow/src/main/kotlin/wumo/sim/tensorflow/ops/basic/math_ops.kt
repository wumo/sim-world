package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.OutputMaker
import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputConvertible
import wumo.sim.tensorflow.ops.gen.gen_array_ops
import wumo.sim.tensorflow.ops.gen.gen_array_ops.transpose
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

fun Output.cast(dataType: DataType<*>): Output =
    tf.cast(this, dataType)

object math_ops {
  private val dtype_hierarchy: Map<DataType<*>, Int> = mapOf(INT32 to 0,
                                                             INT64 to 1,
                                                             FLOAT to 2,
                                                             DOUBLE to 3)
  
  interface API {
    fun abs(x: Output, name: String = "Abs"): Output {
      return gen_math_ops.abs(x, name)
    }
    
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
        else -> gen_math_ops.accumulateNV2(inputs, inferredShape, name)
      }
    }
    
    fun acos(x: Output, name: String = "Acos"): Output {
      return gen_math_ops.acos(x, name)
    }
    
    fun acosh(x: Output, name: String = "Acosh"): Output {
      return gen_math_ops.acosh(x, name)
    }
    
    fun add(x: Output, y: Output, name: String = "Add"): Output {
      return gen_math_ops.add(x, y, name)
    }
    
    fun addN(inputs: List<Output>, name: String = "AddN"): Output {
      return gen_math_ops.addN(inputs, name)
    }
    
    fun addV2(x: Output, y: Output, name: String = "AddV2"): Output {
      return gen_math_ops.addV2(x, y, name)
    }
    
    fun all(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "All"): Output {
      return gen_math_ops.all(input, reductionIndices, keepDims, name)
    }
    
    fun angle(input: Output, tout: DataType<*> = FLOAT, name: String = "Angle"): Output {
      return gen_math_ops.angle(input, tout, name)
    }
    
    fun any(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Any"): Output {
      return gen_math_ops.any(input, reductionIndices, keepDims, name)
    }
    
    fun approximateEqual(x: Output, y: Output, tolerance: Float = 1.0E-5f, name: String = "ApproximateEqual"): Output {
      return gen_math_ops.approximateEqual(x, y, tolerance, name)
    }
    
    fun argmax(a: Output, axis: Int = 0, output_type: DataType<*> = INT64, name: String = "ArgMax") =
        tf.nameScope(name) {
          val dimension = tf.const(axis, "dimension")
          gen_math_ops.argMax(a, dimension, output_type, tf.currentNameScope)
        }
    
    fun argmin(a: Output, axis: Int = 0, output_type: DataType<*> = INT64, name: String = "ArgMin") =
        tf.nameScope(name) {
          val dimension = tf.const(axis, "dimension")
          gen_math_ops.argMin(a, dimension, output_type, tf.currentNameScope)
        }
    
    fun asin(x: Output, name: String = "Asin"): Output {
      return gen_math_ops.asin(x, name)
    }
    
    fun asinh(x: Output, name: String = "Asinh"): Output {
      return gen_math_ops.asinh(x, name)
    }
    
    fun atan(x: Output, name: String = "Atan"): Output {
      return gen_math_ops.atan(x, name)
    }
    
    fun atan2(y: Output, x: Output, name: String = "Atan2"): Output {
      return gen_math_ops.atan2(y, x, name)
    }
    
    fun atanh(x: Output, name: String = "Atanh"): Output {
      return gen_math_ops.atanh(x, name)
    }
    
    fun batchMatMul(x: Output, y: Output, adjX: Boolean = false, adjY: Boolean = false, name: String = "BatchMatMul"): Output {
      return gen_math_ops.batchMatMul(x, y, adjX, adjY, name)
    }
    
    fun besselI0e(x: Output, name: String = "BesselI0e"): Output {
      return gen_math_ops.besselI0e(x, name)
    }
    
    fun besselI1e(x: Output, name: String = "BesselI1e"): Output {
      return gen_math_ops.besselI1e(x, name)
    }
    
    fun betainc(a: Output, b: Output, x: Output, name: String = "Betainc"): Output {
      return gen_math_ops.betainc(a, b, x, name)
    }
    
    fun bincount(arr: Output, size: Output, weights: Output, name: String = "Bincount"): Output {
      return gen_math_ops.bincount(arr, size, weights, name)
    }
    
    fun bucketize(input: Output, boundaries: Array<Float>, name: String = "Bucketize"): Output {
      return gen_math_ops.bucketize(input, boundaries, name)
    }
    
    fun cast(x: Output, dstT: DataType<*>, name: String = "Cast"): Output = run {
      val x = (x as? Variable)?.value ?: x
      if (x.dataType == dstT) x
      else gen_math_ops.cast(x, dstT, name)
    }
    
    fun castArgs(a: Output, b: Output): Pair<Output, Output> {
      val dataType = DataType.mostPrecise(a.dataType, b.dataType)
      val cA = tf.cast(a, dataType)
      val cB = tf.cast(b, dataType)
      return Pair(cA, cB)
    }
    
    fun ceil(x: Output, name: String = "Ceil"): Output {
      return gen_math_ops.ceil(x, name)
    }
    
    fun clipByValue(t: Output, clipValueMin: Output, clipValueMax: Output, name: String = "ClipByValue"): Output {
      return gen_math_ops.clipByValue(t, clipValueMin, clipValueMax, name)
    }
    
    fun compareAndBitpack(input: Output, threshold: Output, name: String = "CompareAndBitpack"): Output {
      return gen_math_ops.compareAndBitpack(input, threshold, name)
    }
    
    fun complex(real: Output, imag: Output, tout: DataType<*> = COMPLEX64, name: String = "Complex"): Output {
      return gen_math_ops.complex(real, imag, tout, name)
    }
    
    fun complexAbs(x: Output, tout: DataType<*> = FLOAT, name: String = "ComplexAbs"): Output {
      return gen_math_ops.complexAbs(x, tout, name)
    }
    
    fun conj(x: Output, name: String = "Conj") =
        when {
          x.dataType.isComplex -> gen_math_ops.conj(x, name)
          x.dataType.isNumeric -> x
          else -> throw IllegalArgumentException("'conjugate' can only take numeric tensors as input.")
        }
    
    fun cos(x: Output, name: String = "Cos"): Output {
      return gen_math_ops.cos(x, name)
    }
    
    fun cosh(x: Output, name: String = "Cosh"): Output {
      return gen_math_ops.cosh(x, name)
    }
    
    fun cross(a: Output, b: Output, name: String = "Cross"): Output {
      return gen_math_ops.cross(a, b, name)
    }
    
    fun cumprod(x: Output, axis: Output, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumprod"): Output {
      return gen_math_ops.cumprod(x, axis, exclusive, reverse, name)
    }
    
    fun cumsum(x: Output, axis: Output, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumsum"): Output {
      return gen_math_ops.cumsum(x, axis, exclusive, reverse, name)
    }
    
    fun digamma(x: Output, name: String = "Digamma"): Output {
      return gen_math_ops.digamma(x, name)
    }
    
    fun div(x: Output, y: Output, name: String = "Div"): Output {
      return gen_math_ops.div(x, y, name)
    }
    
    fun equal(x: Output, y: Output, name: String = "Equal"): Output {
      return gen_math_ops.equal(x, y, name)
    }
    
    fun erf(x: Output, name: String = "Erf"): Output {
      return gen_math_ops.erf(x, name)
    }
    
    fun erfc(x: Output, name: String = "Erfc"): Output {
      return gen_math_ops.erfc(x, name)
    }
    
    fun exp(x: Output, name: String = "Exp"): Output {
      return gen_math_ops.exp(x, name)
    }
    
    fun expm1(x: Output, name: String = "Expm1"): Output {
      return gen_math_ops.expm1(x, name)
    }
    
    fun floor(x: Output, name: String = "Floor"): Output {
      return gen_math_ops.floor(x, name)
    }
    
    fun floorDiv(x: Output, y: Output, name: String = "FloorDiv"): Output {
      return gen_math_ops.floorDiv(x, y, name)
    }
    
    fun floorMod(x: Output, y: Output, name: String = "FloorMod"): Output {
      return gen_math_ops.floorMod(x, y, name)
    }
    
    fun greater(x: Output, y: Output, name: String = "Greater"): Output {
      return gen_math_ops.greater(x, y, name)
    }
    
    fun greaterEqual(a: Output, b: Any, name: String = "GreaterEqual") =
        tf.nameScope(name) {
          val y = tf.const(a.dataType.baseDataType, b, name = "y")
          gen_math_ops.greaterEqual(a, y, tf.currentNameScope)
        }
    
    fun greaterEqual(x: Output, y: Output, name: String = "GreaterEqual"): Output {
      return gen_math_ops.greaterEqual(x, y, name)
    }
    
    fun histogramFixedWidth(values: Output, valueRange: Output, nbins: Output, dtype: DataType<*> = INT32, name: String = "HistogramFixedWidth"): Output {
      return gen_math_ops.histogramFixedWidth(values, valueRange, nbins, dtype, name)
    }
    
    fun igamma(a: Output, x: Output, name: String = "Igamma"): Output {
      return gen_math_ops.igamma(a, x, name)
    }
    
    fun igammaGradA(a: Output, x: Output, name: String = "IgammaGradA"): Output {
      return gen_math_ops.igammaGradA(a, x, name)
    }
    
    fun igammac(a: Output, x: Output, name: String = "Igammac"): Output {
      return gen_math_ops.igammac(a, x, name)
    }
    
    fun imag(input: Output, tout: DataType<*> = FLOAT, name: String = "Imag"): Output {
      return gen_math_ops.imag(input, tout, name)
    }
    
    fun inv(x: Output, name: String = "Inv"): Output {
      return gen_math_ops.inv(x, name)
    }
    
    fun invGrad(y: Output, dy: Output, name: String = "InvGrad"): Output {
      return gen_math_ops.invGrad(y, dy, name)
    }
    
    fun isFinite(x: Output, name: String = "IsFinite"): Output {
      return gen_math_ops.isFinite(x, name)
    }
    
    fun isInf(x: Output, name: String = "IsInf"): Output {
      return gen_math_ops.isInf(x, name)
    }
    
    fun isNan(x: Output, name: String = "IsNan"): Output {
      return gen_math_ops.isNan(x, name)
    }
    
    fun less(x: Output, y: Output, name: String = "Less"): Output {
      return gen_math_ops.less(x, y, name)
    }
    
    fun lessEqual(x: Output, y: Output, name: String = "LessEqual"): Output {
      return gen_math_ops.lessEqual(x, y, name)
    }
    
    fun lgamma(x: Output, name: String = "Lgamma"): Output {
      return gen_math_ops.lgamma(x, name)
    }
    
    fun linSpace(start: Output, stop: Output, num: Output, name: String = "LinSpace"): Output {
      return gen_math_ops.linSpace(start, stop, num, name)
    }
    
    fun log(x: Output, name: String = "Log"): Output {
      return gen_math_ops.log(x, name)
    }
    
    fun log1p(x: Output, name: String = "Log1p"): Output {
      return gen_math_ops.log1p(x, name)
    }
    
    fun logicalAnd(x: Output, y: Output, name: String = "LogicalAnd"): Output {
      return gen_math_ops.logicalAnd(x, y, name)
    }
    
    fun logicalNot(x: Output, name: String = "LogicalNot"): Output {
      return gen_math_ops.logicalNot(x, name)
    }
    
    fun logicalOr(x: Output, y: Output, name: String = "LogicalOr"): Output {
      return gen_math_ops.logicalOr(x, y, name)
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
            gen_math_ops.batchMatMul(x, y, adjointX, adjointY, tf.currentNameScope)
          } else if (cA.dataType == BFLOAT16 || cB.dataType == BFLOAT16 || // "MatMul" does not currently support this type.
              ((aIsSparse || bIsSparse) &&
                  sparseMatMulDataTypes.contains(cA.dataType) &&
                  sparseMatMulDataTypes.contains(cB.dataType))) {
            val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
            val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
            gen_math_ops.sparseMatMul(x, y, transposeX, transposeY, aIsSparse, bIsSparse, tf.currentNameScope)
          } else {
            val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
            val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
            gen_math_ops.matMul(x, y, transposeX, transposeY, tf.currentNameScope)
          }
        }
  
    fun matrixTranspose(a: Output, conjugate: Boolean = false, name: String = "matrix_transpose"): Output =
        tf.nameScope(name) {
        
          val a_shape = a.shape
          val ndims = a_shape.rank
          val perm = if (ndims != -1) {
            require(ndims >= 2) { "Argument 'a' should be a (batch) matrix, with rank >= 2.  Found: $a_shape" }
            tf.const((0 until ndims - 2).asSequence().plus(ndims - 1).plus(ndims - 2).toList().toIntArray())
          } else {
            val a_rank = tf.rank(a)
            val zero = tf.const(0)
            tf.concat(listOf(gen_math_ops.range(zero, a_rank - 2, tf.const(1)),
                             tf.stack(listOf(a_rank - 1, a_rank - 2))),
                      zero)
          }
        
          tf.transpose(a, perm, conjugate)
        }
    
    fun max(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Max"): Output {
      return gen_math_ops.max(input, reductionIndices, keepDims, name)
    }
    
    fun maximum(x: Output, y: Output, name: String = "Maximum"): Output {
      return gen_math_ops.maximum(x, y, name)
    }
    
    fun mean(input: Output, axis: Output? = null, keepDims: Boolean = false, name: String = "mean") =
        gen_math_ops.mean(input, reductionDims(input, axis), keepDims, name)
    
    fun min(input: Output, axis: Output? = null, keepDims: Boolean = false, name: String = "min") =
        gen_math_ops.min(input, reductionDims(input, axis), keepDims, name)
    
    fun minimum(x: Output, y: Output, name: String = "Minimum"): Output {
      return gen_math_ops.minimum(x, y, name)
    }
    
    fun mod(x: Output, y: Output, name: String = "Mod"): Output {
      return gen_math_ops.mod(x, y, name)
    }
    
    fun mul(x: Output, y: Output, name: String = "Mul"): Output {
      return gen_math_ops.mul(x, y, name)
    }
    
    fun neg(x: Output, name: String = "Neg"): Output {
      return gen_math_ops.neg(x, name)
    }
    
    fun notEqual(x: Output, y: Output, name: String = "NotEqual"): Output {
      return gen_math_ops.notEqual(x, y, name)
    }
    
    fun polygamma(a: Output, x: Output, name: String = "Polygamma"): Output {
      return gen_math_ops.polygamma(a, x, name)
    }
    
    fun pow(x: Output, y: Output, name: String = "Pow"): Output {
      return gen_math_ops.pow(x, y, name)
    }
    
    fun prod(input: Output, axis: Output? = null, keepDims: Boolean = false, name: String = "Prod") =
        gen_math_ops.prod(input, reductionDims(input, axis), keepDims, name)
    
    fun quantizeDownAndShrinkRange(input: Output, inputMin: Output, inputMax: Output, outType: DataType<*>, name: String = "QuantizeDownAndShrinkRange"): List<Output> {
      return gen_math_ops.quantizeDownAndShrinkRange(input, inputMin, inputMax, outType, name)
    }
    
    fun quantizedAdd(x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output, toutput: DataType<*> = QINT32, name: String = "QuantizedAdd"): List<Output> {
      return gen_math_ops.quantizedAdd(x, y, minX, maxX, minY, maxY, toutput, name)
    }
    
    fun quantizedMatMul(a: Output, b: Output, minA: Output, maxA: Output, minB: Output, maxB: Output, toutput: DataType<*> = QINT32, transposeA: Boolean = false, transposeB: Boolean = false, tactivation: DataType<*> = QUINT8, name: String = "QuantizedMatMul"): List<Output> {
      return gen_math_ops.quantizedMatMul(a, b, minA, maxA, minB, maxB, toutput, transposeA, transposeB, tactivation, name)
    }
    
    fun quantizedMul(x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output, toutput: DataType<*> = QINT32, name: String = "QuantizedMul"): List<Output> {
      return gen_math_ops.quantizedMul(x, y, minX, maxX, minY, maxY, toutput, name)
    }
  
    fun range(start: OutputMaker, limit: Output, delta: OutputMaker = { tf.const(1, it) }, name: String = "Range"): Output {
      return tf.nameScope(name) {
        val start_t = start("start")
        val delta_t = delta("delta")
        val dtypes = a(start_t.dataType, limit.dataType, delta_t.dataType)
        val inferred_dtype = dtypes.maxBy { dtype_hierarchy[it]!! }!!
        val start = cast(start_t, inferred_dtype)
        val limit = cast(limit, inferred_dtype)
        val delta = cast(delta_t, inferred_dtype)
        gen_math_ops.range(start, limit, delta, tf.currentNameScope)
      }
    }
    
    fun range(start: Output, limit: Output, delta: Output = tf.const(1), name: String = "Range") = run {
      val dtypes = a(start.dataType, limit.dataType, delta.dataType)
      val inferred_dtype = dtypes.maxBy { dtype_hierarchy[it]!! }!!
      val start = cast(start, inferred_dtype)
      val limit = cast(limit, inferred_dtype)
      val delta = cast(delta, inferred_dtype)
      gen_math_ops.range(start, limit, delta, name)
    }
    
    fun real(input: Output, tout: DataType<*> = FLOAT, name: String = "Real"): Output {
      return gen_math_ops.real(input, tout, name)
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
    
    fun realDiv(a: Output, b: Any, name: String = "RealDiv") =
        tf.nameScope("truediv") {
          val y = b as? Output ?: tf.const(a.dataType.baseDataType, b, name = "y")
          gen_math_ops.realDiv(a, y, name = tf.currentNameScope)
        }
    
    fun reciprocal(x: Output, name: String = "Reciprocal"): Output {
      return gen_math_ops.reciprocal(x, name)
    }
    
    fun reciprocalGrad(y: Output, dy: Output, name: String = "ReciprocalGrad"): Output {
      return gen_math_ops.reciprocalGrad(y, dy, name)
    }
    
    fun requantizationRange(input: Output, inputMin: Output, inputMax: Output, name: String = "RequantizationRange"): List<Output> {
      return gen_math_ops.requantizationRange(input, inputMin, inputMax, name)
    }
    
    fun requantize(input: Output, inputMin: Output, inputMax: Output, requestedOutputMin: Output, requestedOutputMax: Output, outType: DataType<*>, name: String = "Requantize"): List<Output> {
      return gen_math_ops.requantize(input, inputMin, inputMax, requestedOutputMin, requestedOutputMax, outType, name)
    }
    
    fun rint(x: Output, name: String = "Rint"): Output {
      return gen_math_ops.rint(x, name)
    }
    
    fun round(x: Output, name: String = "Round"): Output {
      return gen_math_ops.round(x, name)
    }
    
    fun rsqrt(x: Output, name: String = "Rsqrt"): Output {
      return gen_math_ops.rsqrt(x, name)
    }
    
    fun rsqrtGrad(y: Output, dy: Output, name: String = "RsqrtGrad"): Output {
      return gen_math_ops.rsqrtGrad(y, dy, name)
    }
    
    fun segmentMax(data: Output, segmentIds: Output, name: String = "SegmentMax"): Output {
      return gen_math_ops.segmentMax(data, segmentIds, name)
    }
    
    fun segmentMean(data: Output, segmentIds: Output, name: String = "SegmentMean"): Output {
      return gen_math_ops.segmentMean(data, segmentIds, name)
    }
    
    fun segmentMin(data: Output, segmentIds: Output, name: String = "SegmentMin"): Output {
      return gen_math_ops.segmentMin(data, segmentIds, name)
    }
    
    fun segmentProd(data: Output, segmentIds: Output, name: String = "SegmentProd"): Output {
      return gen_math_ops.segmentProd(data, segmentIds, name)
    }
    
    fun segmentSum(data: Output, segmentIds: Output, name: String = "SegmentSum"): Output {
      return gen_math_ops.segmentSum(data, segmentIds, name)
    }
    
    fun select(condition: Output, t: Output, e: Output, name: String = "Select"): Output {
      return gen_math_ops.select(condition, t, e, name)
    }
    
    fun sigmoid(x: Output, name: String = "Sigmoid"): Output {
      return gen_math_ops.sigmoid(x, name)
    }
    
    fun sigmoidGrad(y: Output, dy: Output, name: String = "SigmoidGrad"): Output {
      return gen_math_ops.sigmoidGrad(y, dy, name)
    }
    
    fun sign(x: Output, name: String = "Sign"): Output {
      return gen_math_ops.sign(x, name)
    }
    
    fun sin(x: Output, name: String = "Sin"): Output {
      return gen_math_ops.sin(x, name)
    }
    
    fun sinh(x: Output, name: String = "Sinh"): Output {
      return gen_math_ops.sinh(x, name)
    }
    
    fun sparseMatMul(a: Output, b: Output, transposeA: Boolean = false, transposeB: Boolean = false, aIsSparse: Boolean = false, bIsSparse: Boolean = false, name: String = "SparseMatMul"): Output {
      return gen_math_ops.sparseMatMul(a, b, transposeA, transposeB, aIsSparse, bIsSparse, name)
    }
    
    fun sparseSegmentMean(data: Output, indices: Output, segmentIds: Output, name: String = "SparseSegmentMean"): Output {
      return gen_math_ops.sparseSegmentMean(data, indices, segmentIds, name)
    }
    
    fun sparseSegmentMeanGrad(grad: Output, indices: Output, segmentIds: Output, outputDim0: Output, name: String = "SparseSegmentMeanGrad"): Output {
      return gen_math_ops.sparseSegmentMeanGrad(grad, indices, segmentIds, outputDim0, name)
    }
    
    fun sparseSegmentMeanWithNumSegments(data: Output, indices: Output, segmentIds: Output, numSegments: Output, name: String = "SparseSegmentMeanWithNumSegments"): Output {
      return gen_math_ops.sparseSegmentMeanWithNumSegments(data, indices, segmentIds, numSegments, name)
    }
    
    fun sparseSegmentSqrtN(data: Output, indices: Output, segmentIds: Output, name: String = "SparseSegmentSqrtN"): Output {
      return gen_math_ops.sparseSegmentSqrtN(data, indices, segmentIds, name)
    }
    
    fun sparseSegmentSqrtNGrad(grad: Output, indices: Output, segmentIds: Output, outputDim0: Output, name: String = "SparseSegmentSqrtNGrad"): Output {
      return gen_math_ops.sparseSegmentSqrtNGrad(grad, indices, segmentIds, outputDim0, name)
    }
    
    fun sparseSegmentSqrtNWithNumSegments(data: Output, indices: Output, segmentIds: Output, numSegments: Output, name: String = "SparseSegmentSqrtNWithNumSegments"): Output {
      return gen_math_ops.sparseSegmentSqrtNWithNumSegments(data, indices, segmentIds, numSegments, name)
    }
    
    fun sparseSegmentSum(data: Output, indices: Output, segmentIds: Output, name: String = "SparseSegmentSum"): Output {
      return gen_math_ops.sparseSegmentSum(data, indices, segmentIds, name)
    }
    
    fun sparseSegmentSumWithNumSegments(data: Output, indices: Output, segmentIds: Output, numSegments: Output, name: String = "SparseSegmentSumWithNumSegments"): Output {
      return gen_math_ops.sparseSegmentSumWithNumSegments(data, indices, segmentIds, numSegments, name)
    }
    
    fun sqrt(x: Output, name: String = "Sqrt"): Output {
      return gen_math_ops.sqrt(x, name)
    }
    
    fun sqrtGrad(y: Output, dy: Output, name: String = "SqrtGrad"): Output {
      return gen_math_ops.sqrtGrad(y, dy, name)
    }
    
    fun square(x: Output, name: String = "Square"): Output {
      return gen_math_ops.square(x, name)
    }
    
    fun squaredDifference(x: Output, y: Output, name: String = "SquaredDifference"): Output {
      return gen_math_ops.squaredDifference(x, y, name)
    }
    
    fun sub(x: Output, y: Output, name: String = "Sub"): Output {
      return gen_math_ops.sub(x, y, name)
    }
    
    fun sum(input: Output, axis: Output? = null, keepDims: Boolean = false, name: String = "sum") =
        gen_math_ops.sum(input, reductionDims(input, axis), keepDims, name)
    
    fun tan(x: Output, name: String = "Tan"): Output {
      return gen_math_ops.tan(x, name)
    }
    
    fun tanh(x: Output, name: String = "Tanh"): Output {
      return gen_math_ops.tanh(x, name)
    }
    
    fun tanhGrad(y: Output, dy: Output, name: String = "TanhGrad"): Output {
      return gen_math_ops.tanhGrad(y, dy, name)
    }
    
    fun toFloat(x: Output, name: String = "ToFloat"): Output =
        tf.cast(x, FLOAT, name)
    
    fun toDouble(x: Output, name: String = "ToDouble"): Output =
        tf.cast(x, DOUBLE, name)
    
    fun toInt32(x: Output, name: String = "ToInt32"): Output =
        tf.cast(x, INT32, name)
    
    fun toInt64(x: Output, name: String = "ToInt64"): Output =
        tf.cast(x, INT64, name)
    
    fun toBFloat16(x: Output, name: String = "ToBFloat16"): Output =
        tf.cast(x, BFLOAT16, name)
    
    fun toComplex64(x: Output, name: String = "ToComplex64"): Output =
        tf.cast(x, COMPLEX64, name)
    
    fun toComplex128(x: Output, name: String = "ToComplex128"): Output =
        tf.cast(x, COMPLEX128, name)
    
    fun transpose(input: Output, perm: Output? = null, conjugate: Boolean = false,
                  name: String = "Transpose"): Output {
      val transposeFn: (Output, Output, String) -> Output = if (conjugate && input.dataType.isComplex)
        gen_array_ops::conjugateTranspose
      else
        gen_array_ops::transpose
      return if (perm == null) {
        val rank = gen_array_ops.rank(input)
        val perm = (rank - 1) - gen_math_ops.range(tf.const(0), rank, tf.const(1))
        val transposed = transposeFn(input, perm, name)
        val inputShape = transposed.op.inputs[0].shape
        if (inputShape.rank != -1)
          transposed.setShape(Shape(inputShape.asIntArray()!!.reversedArray()))
        transposed
      } else
        transposeFn(input, perm, name)
    }
    
    fun truncateDiv(x: Output, y: Output, name: String = "TruncateDiv"): Output {
      return gen_math_ops.truncateDiv(x, y, name)
    }
    
    fun truncateMod(x: Output, y: Output, name: String = "TruncateMod"): Output {
      return gen_math_ops.truncateMod(x, y, name)
    }
    
    fun unsortedSegmentMax(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentMax"): Output {
      return gen_math_ops.unsortedSegmentMax(data, segmentIds, numSegments, name)
    }
    
    fun unsortedSegmentMin(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentMin"): Output {
      return gen_math_ops.unsortedSegmentMin(data, segmentIds, numSegments, name)
    }
    
    fun unsortedSegmentProd(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentProd"): Output {
      return gen_math_ops.unsortedSegmentProd(data, segmentIds, numSegments, name)
    }
    
    fun unsortedSegmentSum(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentSum"): Output {
      return gen_math_ops.unsortedSegmentSum(data, segmentIds, numSegments, name)
    }
    
    fun zeta(x: Output, q: Output, name: String = "Zeta"): Output {
      return gen_math_ops.zeta(x, q, name)
    }
    
    private fun reductionDims(x: Output, axis: Output? = null): Output {
      if (axis != null) return axis
      //Fast path: avoid creating Rank and Range ops if ndims is known.
      return tf.const(arange(x.shape.rank), name = "reduction_indices")
      //TODO SparseOutput
      // Otherwise, we rely on Range and Rank to do the right thing at run-time.
      return range(tf.const(0), tf.rank(x))
    }
    
    fun scalarMul(scalar: Output, x: Output, name: String = "scalarMul"): Output {
      return tf.mul(scalar, x)
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