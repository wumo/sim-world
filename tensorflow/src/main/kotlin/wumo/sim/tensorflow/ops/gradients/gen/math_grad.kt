import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.BOOL
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.i
import kotlin.math.PI
import kotlin.math.max
import kotlin.math.sqrt

//
fun register_math_grad() {
  //  /**Gradients for operators defined in math_ops.py.*/
//  /* from__future__importabsolute_import */
///* from__future__importdivision */
///* from__future__importprint_function */
///* importnumpyasnp */
///* fromtensorflow.python.eagerimportcontext */
///* fromtensorflow.python.frameworkimportconstant_op */
///* fromtensorflow.python.frameworkimportdtypes */
///* fromtensorflow.python.frameworkimportops */
///* fromtensorflow.python.frameworkimporttensor_util */
///* fromtensorflow.python.opsimportarray_ops */
///* fromtensorflow.python.opsimportgen_array_ops */
///* fromtensorflow.python.opsimportgen_math_ops */
///* fromtensorflow.python.opsimportmath_ops */
  fun safeShapeDiv(x: Output, y: Output): Output {
    /**Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`.*/
    return tf._floorDiv(x, tf._maximum(y, tf.const(y.dataType, 1)))
  }
  register("ArgMax") { op, grad ->
    /* delop,grad */
    return@register listOf(null, null)
  }
  register("ArgMin") { op, grad ->
    /* delop,grad */
    return@register listOf(null, null)
  }
  fun sumGrad(op: Op, grad: List<OutputLike?>): List<OutputLike?> {
    /**Gradient for Sum.*/
    var grad = grad[0]!!.toOutput()
    val input0Shape = op.inputs[0].shape
    var axes = op.inputs[1]
    val rank = input0Shape.rank
    //TODO Fast path for when reducing to a scalar and rank is known, which adds only reshape and tile ops (and possibly a
    // shape op too).
    if (rank == 0) {
      return listOf(grad, null)
    }
//    if (np.arrayEqual(axes, np.arange(rank))) {
//      if (context.executingEagerly()) {
//        var ctx = context.context()
//        var newShape = ctx.onesRankCache().get(rank)
//        if (newShape == null) {
//          newShape = tf._const(listOf(1) * rank, dtype = dtypes.int32)
//          ctx.onesRankCache().put(rank, newShape)
//
//        }
//
//      } else {
//        var newShape = listOf(1) * rank
//
//      }
//      grad = tf._reshape(grad, newShape)
//      if (null !in input0Shape) {
//        var inputShape = tf._const(input0Shape, dtype = dtypes.int32)
//
//      } else {
//        var inputShape = tf.shape(op.inputs[0])
//
//      }
//      return listOf(tf._tile(grad, inputShape), null)
//    }
    
    val inputShape = tf.shape(op.inputs[0])
    lateinit var outputShapeKeptDims: Output
    lateinit var tileScaling: Output
    tf.colocateWith(inputShape) {
      outputShapeKeptDims = tf.reducedShape(inputShape, op.inputs[1])
      tileScaling = safeShapeDiv(inputShape, outputShapeKeptDims)
    }
    grad = tf._reshape(grad, outputShapeKeptDims)
    return listOf(tf._tile(grad, tileScaling), null)
  }
  
  register("Sum") { op, grad -> sumGrad(op, grad) }
  
  fun minOrMaxGrad(op: Op, grad: List<OutputLike?>): List<OutputLike?> {
    /**Gradient for Min or Max. Amazingly it's precisely the same code.*/
    var grad = grad[0]!!.toOutput()
    val inputShape = tf.shape(op.inputs[0])
    val outputShapeKeptDims = tf.reducedShape(inputShape, op.inputs[1])
    var y = op.outputs[0]
    y = tf._reshape(y, outputShapeKeptDims)
    grad = tf._reshape(grad, outputShapeKeptDims)
    val indicators = tf._cast(tf._equal(y, op.inputs[0]), grad.dataType)
    val numSelected = tf._reshape(tf.sum(indicators, op.inputs[1]), outputShapeKeptDims)
    return listOf(tf._div(indicators, numSelected) * grad, null)
  }
  register("Max") { op, grad ->
    /**Gradient for Max.*/
    return@register minOrMaxGrad(op, grad)
  }
  register("Min") { op, grad ->
    return@register minOrMaxGrad(op, grad)
  }
  register("Mean") { op, grad ->
    /**Gradient for Mean.*/
    val sumGrad = sumGrad(op, grad)[0]!!.toOutput()
    val inputSize = op.inputs[0].size
    val outputSize = op.outputs[0].size
    lateinit var factor: Output
    if (inputSize != -1 && outputSize != -1) {
      factor = tf.const(sumGrad.dataType, inputSize / max(outputSize, 1))
    } else {
      val inputShape = tf.shape(op.inputs[0])
      val outputShape = tf.shape(op.outputs[0])
      var factor = safeShapeDiv(tf.prod(inputShape), tf.prod(outputShape))
    }
    return@register listOf(tf.realDiv(sumGrad, tf.cast(factor, sumGrad.dataType)), null)
  }
  register("Prod") { op, grad ->
    /**Gradient for Prod.*/
    var grad = grad[0]!!.toOutput()
    val inputShape = tf.shape(op.inputs[0])
    var reductionIndices = tf._reshape(op.inputs[1], tf.const(i(-1)))
    val outputShapeKeptDims = tf.reducedShape(inputShape, op.inputs[1])
    val tileScaling = safeShapeDiv(inputShape, outputShapeKeptDims)
    grad = tf._reshape(grad, outputShapeKeptDims)
    grad = tf._tile(grad, tileScaling)
    lateinit var perm: Output
    lateinit var reducedNum: Output
    lateinit var otherNum: Output
    val zero = tf.const(0)
    tf.device("/cpu:0") {
      val rank = tf._rank(op.inputs[0])
      reductionIndices = (reductionIndices + rank) % rank
      val reduced = tf._cast(reductionIndices, INT32)
      val idx = tf.range(zero, rank)
      val (other, _) = tf._listDiff(idx, reduced)
      perm = tf._concatV2(listOf(reduced, other), zero)
      reducedNum = tf.prod(tf.gather(inputShape, reduced))
      otherNum = tf.prod(tf.gather(inputShape, other))
    }
    val permuted = tf._transpose(op.inputs[0], perm)
    val permutedShape = tf.shape(permuted)
    val reshaped = tf._reshape(permuted, tf.stack(listOf(reducedNum, otherNum)))
    val left = tf._cumprod(reshaped, axis = zero, exclusive = true)
    val right = tf._cumprod(reshaped, axis = zero, exclusive = true, reverse = true)
    val y = tf._reshape(tf.conj(left) * tf.conj(right), permutedShape)
    val out = grad * tf._transpose(y, tf._invertPermutation(perm))
    return@register listOf(tf._reshape(out, inputShape), null)
  }
  register("SegmentSum") { op, grad ->
    /**Gradient for SegmentSum.*/
    val grad = grad[0]!!.toOutput()
    return@register listOf(tf.gather(grad, op.inputs[1]), null)
  }
  register("SegmentMean") { op, grad ->
    /**Gradient for SegmentMean.*/
    val grad = grad[0]!!.toOutput()
    val zero = tf.const(0)
    
    val inputRank = tf._rank(op.inputs[0])
    val onesShape = tf._concatV2(listOf(tf.shape(op.inputs[1]),
                                        tf._fill(tf._expandDims(inputRank - 1, zero),
                                                 tf.const(inputRank.dataType, 1))),
                                 zero)
    val ones = tf._fill(onesShape, tf.const(grad.dataType, 1))
    val scaledGrad = tf._div(grad, tf._segmentSum(ones, op.inputs[1]))
    return@register listOf(tf.gather(scaledGrad, op.inputs[1]), null)
  }
  register("SparseSegmentSum") { op, grad ->
    /**Gradient for SparseSegmentSum.*/
    val grad = grad[0]!!.toOutput()
    val inputRows = tf.shape(op.inputs[0])[0]
    return@register listOf(tf._unsortedSegmentSum(tf.gather(grad, op.inputs[2]), op.inputs[1], inputRows), null, null)
  }
  register("SparseSegmentSumWithNumSegments") { op, grad ->
    /**Gradient for SparseSegmentSumWithNumSegments.*/
    val grad = grad[0]!!.toOutput()
    val inputRows = tf.shape(op.inputs[0])[0]
    return@register listOf(tf._unsortedSegmentSum(tf.gather(grad, op.inputs[2]), op.inputs[1], inputRows), null, null, null)
  }
  register("SparseSegmentMean") { op, grad ->
    /**Gradient for SparseSegmentMean.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf._sparseSegmentMeanGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null)
  }
  register("SparseSegmentMeanWithNumSegments") { op, grad ->
    /**Gradient for SparseSegmentMeanWithNumSegments.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf._sparseSegmentMeanGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null, null)
  }
  register("SparseSegmentSqrtN") { op, grad ->
    /**Gradient for SparseSegmentSqrtN.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf._sparseSegmentSqrtNGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null)
  }
  register("SparseSegmentSqrtNWithNumSegments") { op, grad ->
    /**Gradient for SparseSegmentSqrtNWithNumSegments.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf._sparseSegmentSqrtNGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null, null)
  }
  fun segmentMinOrMaxGrad(op: Op, grad: List<OutputLike?>): List<OutputLike?> {
    /** Gradient for SegmentMin and SegmentMax. */
    val grad = grad[0]!!.toOutput()
    val zeros = tf.zerosLike(op.inputs[0], dtype = op.inputs[0].dataType)
    val gatheredOutputs = tf.gather(op.outputs[0], op.inputs[1])
    val isSelected = tf._equal(op.inputs[0], gatheredOutputs)
    val numSelected = tf._segmentSum(tf._cast(isSelected, grad.dataType), op.inputs[1])
    val weightedGrads = tf._div(grad, numSelected)
    val gatheredGrads = tf.gather(weightedGrads, op.inputs[1])
    return listOf(tf.where(isSelected, gatheredGrads, zeros), null)
  }
  register("SegmentMin") { op, grad ->
    /**Gradient for SegmentMin.*/
    return@register segmentMinOrMaxGrad(op, grad)
  }
  register("SegmentMax") { op, grad ->
    /**Gradient for SegmentMax.*/
    return@register segmentMinOrMaxGrad(op, grad)
  }
  fun gatherDropNegatives(params: Output, ids: Output?,
                          zeroClippedIndices: Output? = null,
                          isPositive: Output? = null): List<Output> {
    /** Helper function for unsorted segment ops. Gathers params for
    positive segment ids and gathers 0 for inputs with negative segment id.
    Also returns the clipped indices and a boolean mask with the same shape
    as ids where a positive id is masked as true. With this, the latter two
    can be passed as arguments to this function to reuse them.
     */
    val zeroClippedIndices = zeroClippedIndices ?: tf._maximum(ids!!, tf.zerosLike(ids))
    val gathered = tf.gather(params, zeroClippedIndices)
    val isPositive = isPositive ?: run {
      var isPositive = tf._greaterEqual(ids!!, tf.const(0))
      val minusOne = tf.const(-1)
      repeat(gathered.shape.rank - isPositive.shape.rank - 1) {
        isPositive = tf._expandDims(isPositive, minusOne)
      }
      isPositive and tf.onesLike(gathered, dtype = BOOL)
    }
    val zeroSlice = tf.zerosLike(gathered)
    return listOf(tf.where(isPositive, gathered, zeroSlice), zeroClippedIndices, isPositive)
  }
  
  fun _UnsortedSegmentMinOrMaxGrad(op: Op, grad: List<OutputLike?>): List<OutputLike?> {
    /** Gradient for UnsortedSegmentMin and UnsortedSegmentMax. */
    val grad = grad[0]!!.toOutput()
    val (gatheredOutputs, zeroClippedIndices, isPositive) = gatherDropNegatives(op.outputs[0], op.inputs[1])
    var isSelected = tf._equal(op.inputs[0], gatheredOutputs)
    isSelected = tf._logicalAnd(isSelected, isPositive)
    val numSelected = tf._unsortedSegmentSum(tf._cast(isSelected, grad.dataType), op.inputs[1], op.inputs[2])
    val weightedGrads = tf._div(grad, numSelected)
    val (gatheredGrads, _, _) = gatherDropNegatives(weightedGrads, null, zeroClippedIndices, isPositive)
    val zeros = tf.zerosLike(gatheredGrads)
    return listOf(tf.where(isSelected, gatheredGrads, zeros), null, null)
  }
  register("UnsortedSegmentSum") { op, grad ->
    /**Gradient for UnsortedSegmentSum.*/
    val grad = grad[0]!!.toOutput()
    return@register listOf(gatherDropNegatives(grad, op.inputs[1])[0], null, null)
  }
  register("UnsortedSegmentMax") { op, grad ->
    /** Gradient for UnsortedSegmentMax. */
    return@register _UnsortedSegmentMinOrMaxGrad(op, grad)
  }
  register("UnsortedSegmentMin") { op, grad ->
    /** Gradient for UnsortedSegmentMin. */
    return@register _UnsortedSegmentMinOrMaxGrad(op, grad)
  }
  register("UnsortedSegmentProd") { op, grad ->
    /** Gradient for UnsortedSegmentProd.
    The gradient can be expressed for each segment by dividing the segment's
    product by each element of the segment input tensor, but this approach can't
    deal with zeros in the input.
    Unlike reduce_prod we can't use cumsum here as individual segments may have
    a different number of elements. Therefore we consider three cases:
    1) A segment input contains no zeros and we can safely divide by the input
    tensor.
    2) A segment contains exactly one zero. Then the gradient of each input of
    the segment is zero except for the 0-input, there the gradient is
    the product of the remaining segment entries.
    3) A segment contains at least two zeros. The gradient is zero for all
    segment inputs.
     */
    var grad = grad[0]!!.toOutput()
    val isZero = tf._equal(op.inputs[0], tf.const(0))
    val numZeros = tf._unsortedSegmentSum(tf.cast(isZero, INT32), op.inputs[1], op.inputs[2])
    grad = tf.where(tf._greater(numZeros, tf.const(1)), tf.zerosLike(grad), grad)
    val nonZeroData = tf.where(isZero, tf.onesLike(op.inputs[0]), op.inputs[0])
    val nonZeroProd = tf._unsortedSegmentProd(nonZeroData, op.inputs[1], op.inputs[2])
    val zeroClippedIndices = tf._maximum(op.inputs[1], tf.zerosLike(op.inputs[1]))
    val gatheredProd = tf.gather(op.outputs[0], zeroClippedIndices)
    val gatheredNonZeroProd = tf.gather(nonZeroProd, zeroClippedIndices)
    val prodDividedByEl = gatheredProd / op.inputs[0]
    val partialDerivative = tf.where(isZero, gatheredNonZeroProd, prodDividedByEl)
    val gatheredGrad = gatherDropNegatives(grad, op.inputs[1], zeroClippedIndices)[0]
    return@register listOf(gatheredGrad * partialDerivative, null, null)
  }
  register("Abs") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    return@register listOf(grad * tf._sign(x))
  }
  register("Neg") { op, grad ->
    /**Returns -grad.*/
    val grad = grad[0]!!.toOutput()
    return@register listOf(-grad)
  }
  register("Inv") { op, grad ->
    /**Returns -grad * (1 / x^2).*/
    val grad = grad[0]!!.toOutput()
    val y = op.outputs[0]
    return@register listOf(tf._reciprocalGrad(y, grad))
  }
  register("Reciprocal") { op, grad ->
    /**Returns -grad * (1 / x^2).*/
    val grad = grad[0]!!.toOutput()
    val y = op.outputs[0]
    return@register listOf(tf._reciprocalGrad(y, grad))
  }
  register("InvGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val b = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val ca = tf.conj(op.inputs[0])
      val cg = tf.conj(grad)
      listOf(cg * -2.0 * b * ca, tf._reciprocalGrad(ca, grad))
    }
  }
  register("ReciprocalGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val b = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val ca = tf.conj(op.inputs[0])
      val cg = tf.conj(grad)
      listOf(cg * -2.0 * b * ca, tf._reciprocalGrad(ca, grad))
    }
  }
  register("Square") { op, grad ->
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val y = tf.const(x.dataType, 2.0)
      listOf(tf._mul(grad, tf._mul(x, y)))
    }
  }
  register("Sqrt") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val y = op.outputs[0]
    return@register listOf(tf._sqrtGrad(y, grad))
  }
  register("SqrtGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val a = op.inputs[0]
    val y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      val ga = grad / a
      listOf(-tf.conj(ga) * y, 0.5 * ga)
    }
  }
  register("Rsqrt") { op, grad ->
    /**Returns -0.5 * grad * conj(y)^3.*/
    val grad = grad[0]!!.toOutput()
    val y = op.outputs[0]
    return@register listOf(tf._rsqrtGrad(y, grad))
  }
  register("RsqrtGrad") { op, grad ->
    /**Returns backprop gradient for f(a,b) = -0.5 * b * conj(a)^3.*/
    val grad = grad[0]!!.toOutput()
    val a = op.inputs[0]
    val b = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val ca = tf.conj(a)
      val cg = tf.conj(grad)
      val gradA = -1.5 * cg * b * tf._square(ca)
      val gradB = tf._rsqrtGrad(ca, grad)
      listOf(gradA, gradB)
    }
    
  }
  register("Exp") { op, grad ->
    /**Returns grad * exp(x).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(grad * y)
    }
  }
  register("Expm1") { op, grad ->
    /**Returns grad * exp(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val y = tf._exp(x)
      listOf(grad * y)
      
    }
    
  }
  register("Log") { op, grad ->
    /**Returns grad * (1/x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf._reciprocal(x))
      
    }
    
  }
  register("Log1p") { op, grad ->
    /**Returns grad * (1/(1 + x)).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf._reciprocal(1 + x))
    }
  }
  register("Sinh") { op, grad ->
    /**Returns grad * cosh(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf._cosh(x))
    }
  }
  register("Cosh") { op, grad ->
    /**Returns grad * sinh(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf._sinh(x))
    }
    
  }
  register("Tanh") { op, grad ->
    /**Returns grad * (1 - tanh(x) * tanh(x)).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(tf._tanhGrad(y, grad))
    }
    
  }
  register("Asinh") { op, grad ->
    /**Returns grad * 1/cosh(y).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(grad / tf._cosh(y))
    }
    
  }
  register("Acosh") { op, grad ->
    /**Returns grad * 1/sinh(y).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(grad / tf._sinh(y))
      
    }
    
  }
  register("Atanh") { op, grad ->
    /**Returns grad * 1/ (1 - x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf._square(x)
      val one = tf.const(grad.dataType, 1)
      val inv = tf._reciprocal(tf._sub(one, x2))
      listOf(grad * inv)
    }
  }
  register("TanhGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    return@register tf.controlDependencies(grad) {
      val a = tf.conj(op.inputs[0])
      val b = tf.conj(op.inputs[1])
      listOf(grad * -2.0 * b * a, tf._tanhGrad(a, grad))
    }
  }
  register("Erf") { op, grad ->
    /**Returns grad * 2/sqrt(pi) * exp(-x**2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    val twoOverRootPi = tf.const(grad.dataType, 2 / sqrt(PI))
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * twoOverRootPi * tf._exp(-tf._square(x)))
    }
  }
  register("Erfc") { op, grad ->
    /**Returns -grad * 2/sqrt(pi) * exp(-x**2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    val minusTwoOverRootPi = tf.const(grad.dataType, -2 / sqrt(PI))
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * minusTwoOverRootPi * tf._exp(-tf._square(x)))
    }
  }
  register("Lgamma") { op, grad ->
    /**Returns grad * digamma(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf._digamma(x))
    }
  }
  register("Digamma") { op, grad ->
    /**Compute gradient of the digamma function with respect to its argument.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf._polygamma(tf.const(x.dataType, 1), x))
    }
  }
  register("BesselI0e") { op, grad ->
    /**Compute gradient of bessel_i0e(x) with respect to its argument.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      listOf(grad * listOf(tf._besselI1e(x) - tf._sign(x) * y))
    }
  }
  register("BesselI1e") { op, grad ->
    /**Compute gradient of bessel_i1e(x) with respect to its argument.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      val eps = Math.ulp(1f)//TODO maybe wrong
      val zeros = tf.zerosLike(x)
      val xIsNotTiny = tf._greater(tf._abs(x), tf.const(x.dataType, eps))
      val safeX = tf.where(xIsNotTiny, x, eps + zeros)
      val dyDx = tf._besselI0e(safeX) - y * listOf(tf._sign(safeX) + tf._reciprocal(safeX))
      listOf(grad * tf.where(xIsNotTiny, dyDx, 0.5 + zeros))
    }
  }
  fun IgammaGrad(op: Op, grad: List<OutputLike?>): List<OutputLike?> {
    /**Returns gradient of igamma(a, x) with respect to a and x.*/
    val grad = grad[0]!!.toOutput()
    val a = op.inputs[0]
    val x = op.inputs[1]
    val sa = tf.shape(a)
    val sx = tf.shape(x)
    val (ra, rx) = tf._broadcastGradientArgs(sa, sx)
    return tf.controlDependencies(grad) {
      val partialA = tf._igammaGradA(a, x)
      val partialX = tf._exp(-x + listOf(a - 1) * tf._log(x) - tf._lgamma(a))
      listOf(tf._reshape(tf.sum(partialA * grad, ra), sa), tf._reshape(tf.sum(partialX * grad, rx), sx))
    }
  }
  register("Igamma") { op, grad ->
    IgammaGrad(op, grad)
  }
  register("Igammac") { op, grad ->
    /**Returns gradient of igammac(a, x) = 1 - igamma(a, x) w.r.t. a and x.*/
    val (igammaGradA, igammaGradX) = IgammaGrad(op, grad)
    return@register listOf(-igammaGradA!!.toOutput(), -igammaGradX!!.toOutput())
  }
  register("Betainc") { op, grad ->
    /**Returns gradient of betainc(a, b, x) with respect to x.*/
    val grad = grad[0]!!.toOutput()
    val (a, b, x) = op.inputs
    val sa = tf.shape(a)
    val sx = tf.shape(x)
    val (_, rx) = tf._broadcastGradientArgs(sa, sx)
    val logBeta = listOf(tf._lgamma(a) + tf._lgamma(b) - tf._lgamma(a + b))
    val partialX = tf._exp(listOf(b - 1) * tf._log(1 - x) + listOf(a - 1) * tf._log(x) - logBeta)
    return@register listOf(null, null, tf._reshape(tf.sum(partialX * grad, rx), sx))
  }
  register("Zeta") { op, grad ->
    /**Returns gradient of zeta(x, q) with respect to x and q.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var q = op.inputs[1]
    val sx = tf.shape(x)
    val sq = tf.shape(q)
    val (unusedRx, rq) = tf._broadcastGradientArgs(sx, sq)
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      q = tf.conj(q)
      val partialQ = -x * tf._zeta(x + 1, q)
      listOf(null, tf._reshape(tf.sum(partialQ * grad, rq), sq))
    }
  }
  register("Polygamma") { op, grad ->
    /**Returns gradient of psi(n, x) with respect to n and x.*/
    val grad = grad[0]!!.toOutput()
    var n = op.inputs[0]
    var x = op.inputs[1]
    val sn = tf.shape(n)
    val sx = tf.shape(x)
    val (unusedRn, rx) = tf._broadcastGradientArgs(sn, sx)
    return@register tf.controlDependencies(grad) {
      n = tf.conj(n)
      x = tf.conj(x)
      val partialX = tf._polygamma(n + 1, x)
      listOf(null, tf._reshape(tf.sum(partialX * grad, rx), sx))
    }
    
  }
  register("Sigmoid") { op, grad ->
    /**Returns grad * sigmoid(x) * (1 - sigmoid(x)).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(tf._sigmoidGrad(y, grad))
    }
  }
  register("SigmoidGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    return@register tf.controlDependencies(grad) {
      val a = tf.conj(op.inputs[0])
      val b = tf.conj(op.inputs[1])
      val gb = grad * b
      listOf(gb - 2.0 * gb * a, tf._sigmoidGrad(a, grad))
    }
  }
  register("Sign") { op, grad ->
    /**Returns 0.*/
    var grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    return@register listOf(tf.zeros(tf.shape(x), dtype = x.dataType))
  }
  register("Sin") { op, grad ->
    /**Returns grad * cos(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf._cos(x))
      
    }
    
  }
  register("Cos") { op, grad ->
    /**Returns grad * -sin(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(-grad * tf._sin(x))
    }
  }
  register("Tan") { op, grad ->
    /**Returns grad * 1/sec^2(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val secx = tf._reciprocal(tf._cos(x))
      val secx2 = tf._square(secx)
      listOf(grad * secx2)
    }
  }
  register("Asin") { op, grad ->
    /**Returns grad * 1/sqrt(1-x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf._square(x)
      val one = tf.const(grad.dataType, 1)
      val den = tf._sqrt(tf._sub(one, x2))
      val inv = tf._reciprocal(den)
      listOf(grad * inv)
    }
  }
  register("Acos") { op, grad ->
    /**Returns grad * -1/sqrt(1-x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf._square(x)
      val one = tf.const(grad.dataType, 1)
      val den = tf._sqrt(tf._sub(one, x2))
      val inv = tf._reciprocal(den)
      listOf(-grad * inv)
    }
  }
  register("Atan") { op, grad ->
    /**Returns grad * 1/ (1 + x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf._square(x)
      val one = tf.const(grad.dataType, 1)
      val inv = tf._reciprocal(tf._add(one, x2))
      listOf(grad * inv)
    }
  }
  register("Atan2") { op, grad ->
    /**Returns grad * x / (x^2 + y^2), grad * -y / (x^2 + y^2).*/
    val grad = grad[0]!!.toOutput()
    val y = op.inputs[0]
    val x = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val gradInv = grad / listOf(tf._square(x) + tf._square(y))
      listOf(x * gradInv, -y * gradInv)
    }
  }
  register("AddN") { op, grad ->
    /**Copies the gradient to all inputs.*/
    val grad = grad[0]!!.toOutput()
    return@register List(op.numInputs) { grad }
  }
  fun shapesFullySpecifiedAndEqual(x: Output, y: Output, grad: Output): Boolean {
    val xShape = x.shape
    val yShape = y.shape
    val gradShape = grad.shape
    return xShape == yShape && xShape == gradShape &&
        xShape.isFullyDefined && yShape.isFullyDefined && gradShape.isFullyDefined
  }
  register("Add") { op, grad ->
    /**Gradient for Add.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.inputs[1]
    if (shapesFullySpecifiedAndEqual(x, y, grad)) {
      return@register listOf(grad, grad)
    }
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    return@register listOf(tf._reshape(tf.sum(grad, rx), sx), tf._reshape(tf.sum(grad, ry), sy))
  }
  register("Sub") { op, grad ->
    /**Gradient for Sub.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.inputs[1]
    if (shapesFullySpecifiedAndEqual(x, y, grad)) {
      return@register listOf(grad, -grad)
    }
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    return@register listOf(tf._reshape(tf.sum(grad, rx), sx), tf._reshape(-tf.sum(grad, ry), sy))
  }
  register("Mul") { op, grad ->
    /**The gradient of scalar multiplication.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    if (shapesFullySpecifiedAndEqual(x, y, grad) && (grad.dataType == INT32 || grad.dataType == FLOAT)) {
      return@register listOf(tf._mul(grad, y), tf._mul(grad, x))
    }
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    return@register listOf(tf._reshape(tf.sum(tf._mul(grad, y), rx), sx), tf._reshape(tf.sum(tf._mul(x, grad), ry), sy))
  }
  register("Div") { op, grad ->
    /**The gradient for the Div operator.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    return@register listOf(tf._reshape(tf.sum(tf._div(grad, y), rx), sx), tf._reshape(tf.sum(grad * tf._div(tf._div(-x, y), y), ry), sy))
  }
  register("FloorDiv") { op, grad ->
    /**The gradient for the FloorDiv operator.*/
    var grad = grad[0]!!.toOutput()
    return@register listOf(null, null)
  }
  register("FloorMod") { op, grad ->
    /**Returns grad * (1, -floor(x/y)).*/
    val grad = grad[0]!!.toOutput()
    val x = tf.conj(op.inputs[0])
    val y = tf.conj(op.inputs[1])
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    val floorXy = tf._floorDiv(x, y)
    val gx = tf._reshape(tf.sum(grad, rx), sx)
    val gy = tf._reshape(tf.sum(grad * tf._neg(floorXy), ry), sy)
    return@register listOf(gx, gy)
  }
  register("TruncateDiv") { op, grad ->
    var grad = grad[0]!!.toOutput()
    return@register listOf(null, null)
  }
  register("RealDiv") { op, grad ->
    /**RealDiv op gradient.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    return@register listOf(tf._reshape(tf.sum(tf._realDiv(grad, y), rx), sx),
                           tf._reshape(tf.sum(grad * tf._realDiv(tf._realDiv(-x, y), y), ry), sy))
  }
  register("DivNoNan") { op, grad ->
    /**DivNoNan op gradient.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    return@register listOf(tf._reshape(tf.sum(tf._divNoNan(grad, y), rx), sx),
                           tf._reshape(tf.sum(grad * tf._divNoNan(tf._divNoNan(-x, y), y), ry), sy))
  }
  register("Pow") { op, grad ->
    /**Returns grad * (y*x^(y-1), z*log(x)).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    var z = op.outputs[0]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    z = tf.conj(z)
    val gx = tf._reshape(tf.sum(grad * y * tf._pow(x, y - 1), rx), sx)
    val logX = if (x.dataType.isComplex)
      tf.where(tf._notEqual(x, tf.const(0)), tf._log(x), tf.zerosLike(x))
    else
      tf.where(tf._greater(x, tf.const(0)), tf._log(x), tf.zerosLike(x))
    val gy = tf._reshape(tf.sum(grad * z * logX, ry), sy)
    return@register listOf(gx, gy)
  }
  fun maximumMinimumGrad(op: Op, grad: List<OutputLike?>, selectorOp: (Output, Output) -> Output): List<OutputLike?> {
    /**Factor out the code for the gradient of Maximum or Minimum.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.inputs[1]
    val gdtype = grad.dataType
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val gradshape = tf.shape(grad)
    val zeros = tf.zeros(gradshape, gdtype)
    val xmask = selectorOp(x, y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    val xgrad = tf.where(xmask, grad, zeros)
    val ygrad = tf.where(xmask, zeros, grad)
    val gx = tf._reshape(tf.sum(xgrad, rx), sx)
    val gy = tf._reshape(tf.sum(ygrad, ry), sy)
    return listOf(gx, gy)
  }
  register("Maximum") { op, grad ->
    /**Returns grad*(x > y, x <= y) with type of grad.*/
    return@register maximumMinimumGrad(op, grad) { x, y -> tf._greaterEqual(x, y) }
  }
  register("Minimum") { op, grad ->
    /**Returns grad*(x < y, x >= y) with type of grad.*/
    return@register maximumMinimumGrad(op, grad) { x, y -> tf._lessEqual(x, y) }
  }
  register("SquaredDifference") { op, grad ->
    /**Returns the gradient for (x-y)^2.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.inputs[1]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    val xGrad = tf.controlDependencies(grad) {
      tf.scalarMul(tf.const(grad.dataType, 2.0), grad) * (x - y)
    }
    return@register listOf(tf._reshape(tf.sum(xGrad, rx), sx),
                           -tf._reshape(tf.sum(xGrad, ry), sy))
  }
  registerNonDifferentiable("Less")
  registerNonDifferentiable("LessEqual")
  registerNonDifferentiable("Greater")
  registerNonDifferentiable("GreaterEqual")
  registerNonDifferentiable("Equal")
  registerNonDifferentiable("ApproximateEqual")
  registerNonDifferentiable("NotEqual")
  registerNonDifferentiable("LogicalAnd")
  registerNonDifferentiable("LogicalOr")
  registerNonDifferentiable("LogicalNot")
  register("Select") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val c = op.inputs[0]
    val x = op.inputs[1]
    val zeros = tf.zerosLike(x)
    return@register listOf(null,
                           tf.where(c, grad, zeros),
                           tf.where(c, zeros, grad))
  }
  register("MatMul") { op, grad ->
    /**Gradient for MatMul.*/
    val grad = grad[0]!!.toOutput()
    val tA = op.attrBool("transpose_a")
    val tB = op.attrBool("transpose_b")
    val a = tf.conj(op.inputs[0])
    val b = tf.conj(op.inputs[1])
    lateinit var gradA: Output
    lateinit var gradB: Output
    when {
      !tA && !tB -> {
        gradA = tf._matMul(grad, b, transpose_b = true)
        gradB = tf._matMul(a, grad, transpose_a = true)
      }
      !tA && tB -> {
        gradA = tf._matMul(grad, b)
        gradB = tf._matMul(grad, a, transpose_a = true)
      }
      tA && !tB -> {
        gradA = tf._matMul(b, grad, transpose_b = true)
        gradB = tf._matMul(a, grad)
      }
      tA && tB -> {
        gradA = tf._matMul(b, grad, transpose_a = true, transpose_b = true)
        gradB = tf._matMul(grad, a, transpose_a = true, transpose_b = true)
      }
    }
    return@register listOf(gradA, gradB)
  }
  register("SparseMatMul") { op, grad ->
    /**Gradient for SparseMatMul.*/
    val grad = grad[0]!!.toOutput()
    val tA = op.attrBool("transpose_a")
    val tB = op.attrBool("transpose_b")
    val isSparse = mapOf(
        op.inputs[0] to op.attrBool("a_is_sparse"),
        op.inputs[1] to op.attrBool("b_is_sparse"),
        grad to (grad.op.opType == "ReluGrad"))
    
    fun sparseMatMul(t1: Output, t2: Output, outDtype: DataType<*>,
                     transposeA: Boolean = false, transposeB: Boolean = false): Output {
      /**Helper function to create SparseMatMul op.*/
      val t1Sparse = isSparse[t1]!!
      val t2Sparse = isSparse[t2]!!
      var transposeB = transposeB
      var t2 = t2
      if (transposeB) {
        t2 = tf.transpose(t2)
        transposeB = false
      }
      var prod = tf.matMul(t1, t2,
                           transposeA = transposeA,
                           transposeB = transposeB,
                           aIsSparse = t1Sparse, bIsSparse = t2Sparse)
      if (prod.dataType != outDtype)
        prod = tf._cast(prod, outDtype)
      return prod
    }
    
    val dtypeA = op.inputs[0].dataType
    val dtypeB = op.inputs[1].dataType
    return@register when {
      !tA && !tB -> {
        listOf(sparseMatMul(grad, op.inputs[1], dtypeA, transposeB = true), sparseMatMul(op.inputs[0], grad, dtypeB, transposeA = true))
      }
      !tA && tB -> {
        listOf(sparseMatMul(grad, op.inputs[1], dtypeA), sparseMatMul(grad, op.inputs[0], dtypeB, transposeA = true))
      }
      tA && !tB -> {
        listOf(sparseMatMul(op.inputs[1], grad, dtypeA, transposeB = true), sparseMatMul(op.inputs[0], grad, dtypeB))
      }
      tA && tB -> {
        listOf(sparseMatMul(op.inputs[1], grad, dtypeA, transposeA = true, transposeB = true), sparseMatMul(grad, op.inputs[0], dtypeB, transposeA = true, transposeB = true))
      }
      else -> error("")
    }
    
  }
  register("Floor") { op, grad ->
    return@register listOf(null)
  }
  register("Ceil") { op, grad ->
    return@register listOf(null)
  }
  register("Round") { op, grad ->
    return@register listOf(null)
  }
  register("Rint") { op, grad ->
    return@register listOf(null)
  }
  register("BatchMatMul") { op, grad ->
    /**Returns the gradient of x and y given the gradient of x * y.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.inputs[1]
    val adjX = op.attrBool("adj_x")
    val adjY = op.attrBool("adj_y")
    lateinit var gradX: Output
    lateinit var gradY: Output
    if (!adjX) {
      if (!adjY) {
        gradX = tf.matMul(grad, y, conjugateA = false, conjugateB = true)
        gradY = tf.matMul(x, grad, conjugateA = true, conjugateB = false)
      } else {
        gradX = tf.matMul(grad, y, conjugateA = false, conjugateB = false)
        gradY = tf.matMul(grad, x, conjugateA = true, conjugateB = false)
      }
    } else {
      if (!adjY) {
        gradX = tf.matMul(y, grad, conjugateA = false, conjugateB = true)
        gradY = tf.matMul(x, grad, conjugateA = false, conjugateB = false)
      } else {
        gradX = tf.matMul(y, grad, conjugateA = true, conjugateB = true)
        gradY = tf.matMul(grad, x, conjugateA = true, conjugateB = true)
      }
    }
    return@register listOf(gradX, gradY)
  }
  registerNonDifferentiable("Range")
  registerNonDifferentiable("LinSpace")
  register("Complex") { op, grad ->
    /**Returns the real and imaginary components of 'grad', respectively.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.inputs[1]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf._broadcastGradientArgs(sx, sy)
    return@register listOf(tf._reshape(tf.sum(tf._real(grad), rx), sx), tf._reshape(tf.sum(tf._imag(grad), ry), sy))
  }
  register("Real") { op, grad ->
    /**Returns 'grad' as the real part and set the imaginary part 0.*/
    val grad = grad[0]!!.toOutput()
    val zero = tf.const(grad.dataType, 0)
    return@register listOf(tf._complex(grad, zero))
  }
  register("Imag") { op, grad ->
    /**Returns 'grad' as the imaginary part and set the real part 0.*/
    val grad = grad[0]!!.toOutput()
    val zero = tf.const(grad.dataType, 0)
    return@register listOf(tf._complex(zero, grad))
  }
  register("Angle") { op, grad ->
    /**Returns -grad / (Im(x) + iRe(x))*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      val re = tf._real(x)
      val im = tf._imag(x)
      val z = tf._reciprocal(tf._complex(im, re))
      val zero = tf.const(grad.dataType, 0)
      val complexGrad = tf._complex(grad, zero)
      listOf(-complexGrad * z)
    }
  }
  register("Conj") { op, grad ->
    /**Returns the complex conjugate of grad.*/
    val grad = grad[0]!!.toOutput()
    return@register listOf(tf.conj(grad))
  }
  register("ComplexAbs") { op, grad ->
    /**Returns the gradient of ComplexAbs.*/
    val grad = grad[0]!!.toOutput()
    return@register listOf(tf._complex(grad, tf.zerosLike(grad)) * tf._sign(op.inputs[0]))
  }
  register("Cast") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val srcType = op.inputs[0].dataType.baseDataType
    val dstType = grad.dataType.baseDataType
    return@register if (srcType.isFloatOrComplex && dstType.isFloatOrComplex)
      listOf(tf.cast(grad, srcType))
    else
      listOf(null)
  }
  register("Cross") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val u = op.inputs[0]
    val v = op.inputs[1]
    return@register listOf(tf._cross(v, grad), tf._cross(grad, u))
  }
  register("Cumsum") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val axis = op.inputs[1]
    val exclusive = op.attrBool("exclusive")
    val reverse = op.attrBool("reverse")
    return@register listOf(tf._cumsum(grad, axis, exclusive = exclusive, reverse = !reverse), null)
  }
  register("Cumprod") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val axis = op.inputs[1]
    val exclusive = op.attrBool("exclusive")
    val reverse = op.attrBool("reverse")
    val prod = tf._cumprod(x, axis, exclusive = exclusive, reverse = reverse)
    val out = tf._cumsum(prod * grad, axis, exclusive = exclusive, reverse = !reverse)
    return@register listOf(out / x, null)
  }
}