import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.basic.*
import wumo.sim.tensorflow.ops.gen.gen_array_ops
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.registerNonDifferentiable
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.BOOL
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.i
import wumo.sim.util.ndarray.NDArray.Companion.toNDArray
import wumo.sim.util.ndarray.arrayEqual
import kotlin.math.PI
import kotlin.math.max
import kotlin.math.sqrt

//
fun register_math_grad() {
  //  /**Gradients for operators defined in math_ops.py.*/
  fun safeShapeDiv(x: Output, y: Output): Output {
    /**Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`.*/
    return tf.floorDiv(x, tf.maximum(y, tf.const(y.dataType, 1)))
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
    val input = op.inputs[0]
    val input0Shape = input.shape
    val rank = input0Shape.rank
    //TODO Fast path for when reducing to a scalar and rank is known, which adds only reshape and tile ops (and possibly a
    // shape op too).
    if (rank == 0)
      return listOf(grad, null)
    val axes = constantValue(op.inputs[1])
    if (axes != null && arrayEqual(axes, toNDArray((0 until rank).toList()))) {
      // In this case the reduction was over all dimensions.
      val newShape = IntArray(rank) { 1 }
      
      grad = tf.reshape(grad, tf.const(newShape))
      val inputshape = if (input0Shape.isFullyDefined)
      // If the shape is not fully defined but the rank is, we use the shape op.
        input0Shape.toOutput()
      else
        tf.shape(input)
      return listOf(tf.tile(grad, inputshape), null)
    }
    
    val inputShape = tf.shape(input)
    lateinit var outputShapeKeptDims: Output
    lateinit var tileScaling: Output
    tf.colocateWith(inputShape) {
      outputShapeKeptDims = tf.reducedShape(inputShape, op.inputs[1])
      tileScaling = safeShapeDiv(inputShape, outputShapeKeptDims)
    }
    grad = tf.reshape(grad, outputShapeKeptDims)
    return listOf(tf.tile(grad, tileScaling), null)
  }
  
  register("Sum") { op, grad -> sumGrad(op, grad) }
  
  fun minOrMaxGrad(op: Op, grad: List<OutputLike?>): List<OutputLike?> {
    /**Gradient for Min or Max. Amazingly it's precisely the same code.*/
    var grad = grad[0]!!.toOutput()
    val inputShape = tf.shape(op.inputs[0])
    val outputShapeKeptDims = tf.reducedShape(inputShape, op.inputs[1])
    var y = op.outputs[0]
    y = tf.reshape(y, outputShapeKeptDims)
    grad = tf.reshape(grad, outputShapeKeptDims)
    val indicators = tf.cast(tf.equal(y, op.inputs[0]), grad.dataType)
    val numSelected = tf.reshape(tf.sum(indicators, op.inputs[1]), outputShapeKeptDims)
    return listOf(tf.div(indicators, numSelected) * grad, null)
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
    val inputShape = op.inputs[0].shape
    val outputShape = op.outputs[0].shape
    lateinit var factor: Output
    factor = if (inputShape.isFullyDefined &&
        outputShape.isFullyDefined) {
      val inputSize=inputShape.numElements()
      val outputSize=outputShape.numElements()
      tf.const(sumGrad.dataType, inputSize / max(outputSize, 1))
    } else {
      val inputShape = tf.shape(op.inputs[0])
      val outputShape = tf.shape(op.outputs[0])
      safeShapeDiv(tf.prod(inputShape), tf.prod(outputShape))
    }
    return@register listOf(tf.realDiv(sumGrad, tf.cast(factor, sumGrad.dataType)), null)
  }
  register("Prod") { op, grad ->
    /**Gradient for Prod.*/
    var grad = grad[0]!!.toOutput()
    val inputShape = tf.shape(op.inputs[0])
    var reductionIndices = tf.reshape(op.inputs[1], tf.const(i(-1)))
    val outputShapeKeptDims = tf.reducedShape(inputShape, op.inputs[1])
    val tileScaling = safeShapeDiv(inputShape, outputShapeKeptDims)
    grad = tf.reshape(grad, outputShapeKeptDims)
    grad = tf.tile(grad, tileScaling)
    lateinit var perm: Output
    lateinit var reducedNum: Output
    lateinit var otherNum: Output
    val zero = tf.const(0)
    tf.device("/cpu:0") {
      val rank = tf.rank(op.inputs[0])
      reductionIndices = (reductionIndices + rank) % rank
      val reduced = tf.cast(reductionIndices, INT32)
      val idx = tf.range(zero, rank)
      val (other, _) = tf.listDiff(idx, reduced)
      perm = tf.concat(listOf(reduced, other), zero)
      reducedNum = tf.prod(tf.gather(inputShape, reduced))
      otherNum = tf.prod(tf.gather(inputShape, other))
    }
    val permuted = tf.transpose(op.inputs[0], perm)
    val permutedShape = tf.shape(permuted)
    val reshaped = tf.reshape(permuted, tf.stack(listOf(reducedNum, otherNum)))
    val left = tf.cumprod(reshaped, axis = zero, exclusive = true)
    val right = tf.cumprod(reshaped, axis = zero, exclusive = true, reverse = true)
    val y = tf.reshape(tf.conj(left) * tf.conj(right), permutedShape)
    val out = grad * tf.transpose(y, tf.invertPermutation(perm))
    return@register listOf(tf.reshape(out, inputShape), null)
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
    
    val inputRank = tf.rank(op.inputs[0])
    val onesShape = tf.concat(listOf(tf.shape(op.inputs[1]),
                                     tf.fill(tf.expandDims(inputRank - 1, zero),
                                             tf.const(inputRank.dataType, 1))),
                              zero)
    val ones = tf.fill(onesShape, tf.const(grad.dataType, 1))
    val scaledGrad = tf.div(grad, tf.segmentSum(ones, op.inputs[1]))
    return@register listOf(tf.gather(scaledGrad, op.inputs[1]), null)
  }
  register("SparseSegmentSum") { op, grad ->
    /**Gradient for SparseSegmentSum.*/
    val grad = grad[0]!!.toOutput()
    val inputRows = tf.shape(op.inputs[0])[0]
    return@register listOf(tf.unsortedSegmentSum(tf.gather(grad, op.inputs[2]), op.inputs[1], inputRows), null, null)
  }
  register("SparseSegmentSumWithNumSegments") { op, grad ->
    /**Gradient for SparseSegmentSumWithNumSegments.*/
    val grad = grad[0]!!.toOutput()
    val inputRows = tf.shape(op.inputs[0])[0]
    return@register listOf(tf.unsortedSegmentSum(tf.gather(grad, op.inputs[2]), op.inputs[1], inputRows), null, null, null)
  }
  register("SparseSegmentMean") { op, grad ->
    /**Gradient for SparseSegmentMean.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf.sparseSegmentMeanGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null)
  }
  register("SparseSegmentMeanWithNumSegments") { op, grad ->
    /**Gradient for SparseSegmentMeanWithNumSegments.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf.sparseSegmentMeanGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null, null)
  }
  register("SparseSegmentSqrtN") { op, grad ->
    /**Gradient for SparseSegmentSqrtN.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf.sparseSegmentSqrtNGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null)
  }
  register("SparseSegmentSqrtNWithNumSegments") { op, grad ->
    /**Gradient for SparseSegmentSqrtNWithNumSegments.*/
    val grad = grad[0]!!.toOutput()
    val dim0 = tf.shape(op.inputs[0])[0]
    return@register listOf(tf.sparseSegmentSqrtNGrad(grad, op.inputs[1], op.inputs[2], dim0), null, null, null)
  }
  fun segmentMinOrMaxGrad(op: Op, grad: List<OutputLike?>): List<OutputLike?> {
    /** Gradient for SegmentMin and SegmentMax. */
    val grad = grad[0]!!.toOutput()
    val zeros = tf.zerosLike(op.inputs[0], dtype = op.inputs[0].dataType)
    val gatheredOutputs = tf.gather(op.outputs[0], op.inputs[1])
    val isSelected = tf.equal(op.inputs[0], gatheredOutputs)
    val numSelected = tf.segmentSum(tf.cast(isSelected, grad.dataType), op.inputs[1])
    val weightedGrads = tf.div(grad, numSelected)
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
    val zeroClippedIndices = zeroClippedIndices ?: tf.maximum(ids!!, tf.zerosLike(ids))
    val gathered = tf.gather(params, zeroClippedIndices)
    val isPositive = isPositive ?: run {
      var isPositive = tf.greaterEqual(ids!!, tf.const(0))
      val minusOne = tf.const(-1)
      repeat(gathered.shape.rank - isPositive.shape.rank - 1) {
        isPositive = tf.expandDims(isPositive, minusOne)
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
    var isSelected = tf.equal(op.inputs[0], gatheredOutputs)
    isSelected = tf.logicalAnd(isSelected, isPositive)
    val numSelected = tf.unsortedSegmentSum(tf.cast(isSelected, grad.dataType), op.inputs[1], op.inputs[2])
    val weightedGrads = tf.div(grad, numSelected)
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
    val isZero = tf.equal(op.inputs[0], tf.const(0))
    val numZeros = tf.unsortedSegmentSum(tf.cast(isZero, INT32), op.inputs[1], op.inputs[2])
    grad = tf.where(tf.greater(numZeros, tf.const(1)), tf.zerosLike(grad), grad)
    val nonZeroData = tf.where(isZero, tf.onesLike(op.inputs[0]), op.inputs[0])
    val nonZeroProd = tf.unsortedSegmentProd(nonZeroData, op.inputs[1], op.inputs[2])
    val zeroClippedIndices = tf.maximum(op.inputs[1], tf.zerosLike(op.inputs[1]))
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
    return@register listOf(grad * tf.sign(x))
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
    return@register listOf(tf.reciprocalGrad(y, grad))
  }
  register("Reciprocal") { op, grad ->
    /**Returns -grad * (1 / x^2).*/
    val grad = grad[0]!!.toOutput()
    val y = op.outputs[0]
    return@register listOf(tf.reciprocalGrad(y, grad))
  }
  register("InvGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val b = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val ca = tf.conj(op.inputs[0])
      val cg = tf.conj(grad)
      listOf(cg * -2.0 * b * ca, tf.reciprocalGrad(ca, grad))
    }
  }
  register("ReciprocalGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val b = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val ca = tf.conj(op.inputs[0])
      val cg = tf.conj(grad)
      listOf(cg * -2.0 * b * ca, tf.reciprocalGrad(ca, grad))
    }
  }
  register("Square") { op, grad ->
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val y = tf.const(x.dataType, 2.0)
      listOf(tf.mul(grad, tf.mul(x, y)))
    }
  }
  register("Sqrt") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val y = op.outputs[0]
    return@register listOf(tf.sqrtGrad(y, grad))
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
    return@register listOf(tf.rsqrtGrad(y, grad))
  }
  register("RsqrtGrad") { op, grad ->
    /**Returns backprop gradient for f(a,b) = -0.5 * b * conj(a)^3.*/
    val grad = grad[0]!!.toOutput()
    val a = op.inputs[0]
    val b = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val ca = tf.conj(a)
      val cg = tf.conj(grad)
      val gradA = -1.5 * cg * b * tf.square(ca)
      val gradB = tf.rsqrtGrad(ca, grad)
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
      val y = tf.exp(x)
      listOf(grad * y)
      
    }
    
  }
  register("Log") { op, grad ->
    /**Returns grad * (1/x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf.reciprocal(x))
      
    }
    
  }
  register("Log1p") { op, grad ->
    /**Returns grad * (1/(1 + x)).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf.reciprocal(1 + x))
    }
  }
  register("Sinh") { op, grad ->
    /**Returns grad * cosh(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf.cosh(x))
    }
  }
  register("Cosh") { op, grad ->
    /**Returns grad * sinh(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf.sinh(x))
    }
    
  }
  register("Tanh") { op, grad ->
    /**Returns grad * (1 - tanh(x) * tanh(x)).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(tf.tanhGrad(y, grad))
    }
    
  }
  register("Asinh") { op, grad ->
    /**Returns grad * 1/cosh(y).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(grad / tf.cosh(y))
    }
    
  }
  register("Acosh") { op, grad ->
    /**Returns grad * 1/sinh(y).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(grad / tf.sinh(y))
      
    }
    
  }
  register("Atanh") { op, grad ->
    /**Returns grad * 1/ (1 - x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf.square(x)
      val one = tf.const(grad.dataType, 1)
      val inv = tf.reciprocal(tf.sub(one, x2))
      listOf(grad * inv)
    }
  }
  register("TanhGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    return@register tf.controlDependencies(grad) {
      val a = tf.conj(op.inputs[0])
      val b = tf.conj(op.inputs[1])
      listOf(grad * -2.0 * b * a, tf.tanhGrad(a, grad))
    }
  }
  register("Erf") { op, grad ->
    /**Returns grad * 2/sqrt(pi) * exp(-x**2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    val twoOverRootPi = tf.const(grad.dataType, 2 / sqrt(PI))
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * twoOverRootPi * tf.exp(-tf.square(x)))
    }
  }
  register("Erfc") { op, grad ->
    /**Returns -grad * 2/sqrt(pi) * exp(-x**2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    val minusTwoOverRootPi = tf.const(grad.dataType, -2 / sqrt(PI))
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * minusTwoOverRootPi * tf.exp(-tf.square(x)))
    }
  }
  register("Lgamma") { op, grad ->
    /**Returns grad * digamma(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf.digamma(x))
    }
  }
  register("Digamma") { op, grad ->
    /**Compute gradient of the digamma function with respect to its argument.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(grad * tf.polygamma(tf.const(x.dataType, 1), x))
    }
  }
  register("BesselI0e") { op, grad ->
    /**Compute gradient of bessel_i0e(x) with respect to its argument.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      listOf(grad * listOf(tf.besselI1e(x) - tf.sign(x) * y))
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
      val xIsNotTiny = tf.greater(tf.abs(x), tf.const(x.dataType, eps))
      val safeX = tf.where(xIsNotTiny, x, eps + zeros)
      val dyDx = tf.besselI0e(safeX) - y * listOf(tf.sign(safeX) + tf.reciprocal(safeX))
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
    val (ra, rx) = tf.broadcastGradientArgs(sa, sx)
    return tf.controlDependencies(grad) {
      val partialA = tf.igammaGradA(a, x)
      val partialX = tf.exp(-x + listOf(a - 1) * tf.log(x) - tf.lgamma(a))
      listOf(tf.reshape(tf.sum(partialA * grad, ra), sa), tf.reshape(tf.sum(partialX * grad, rx), sx))
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
    val (_, rx) = tf.broadcastGradientArgs(sa, sx)
    val logBeta = listOf(tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a + b))
    val partialX = tf.exp(listOf(b - 1) * tf.log(1 - x) + listOf(a - 1) * tf.log(x) - logBeta)
    return@register listOf(null, null, tf.reshape(tf.sum(partialX * grad, rx), sx))
  }
  register("Zeta") { op, grad ->
    /**Returns gradient of zeta(x, q) with respect to x and q.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var q = op.inputs[1]
    val sx = tf.shape(x)
    val sq = tf.shape(q)
    val (unusedRx, rq) = tf.broadcastGradientArgs(sx, sq)
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      q = tf.conj(q)
      val partialQ = -x * tf.zeta(x + 1, q)
      listOf(null, tf.reshape(tf.sum(partialQ * grad, rq), sq))
    }
  }
  register("Polygamma") { op, grad ->
    /**Returns gradient of psi(n, x) with respect to n and x.*/
    val grad = grad[0]!!.toOutput()
    var n = op.inputs[0]
    var x = op.inputs[1]
    val sn = tf.shape(n)
    val sx = tf.shape(x)
    val (unusedRn, rx) = tf.broadcastGradientArgs(sn, sx)
    return@register tf.controlDependencies(grad) {
      n = tf.conj(n)
      x = tf.conj(x)
      val partialX = tf.polygamma(n + 1, x)
      listOf(null, tf.reshape(tf.sum(partialX * grad, rx), sx))
    }
    
  }
  register("Sigmoid") { op, grad ->
    /**Returns grad * sigmoid(x) * (1 - sigmoid(x)).*/
    val grad = grad[0]!!.toOutput()
    var y = op.outputs[0]
    return@register tf.controlDependencies(grad) {
      y = tf.conj(y)
      listOf(tf.sigmoidGrad(y, grad))
    }
  }
  register("SigmoidGrad") { op, grad ->
    val grad = grad[0]!!.toOutput()
    return@register tf.controlDependencies(grad) {
      val a = tf.conj(op.inputs[0])
      val b = tf.conj(op.inputs[1])
      val gb = grad * b
      listOf(gb - 2.0 * gb * a, tf.sigmoidGrad(a, grad))
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
      listOf(grad * tf.cos(x))
      
    }
    
  }
  register("Cos") { op, grad ->
    /**Returns grad * -sin(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      listOf(-grad * tf.sin(x))
    }
  }
  register("Tan") { op, grad ->
    /**Returns grad * 1/sec^2(x).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val secx = tf.reciprocal(tf.cos(x))
      val secx2 = tf.square(secx)
      listOf(grad * secx2)
    }
  }
  register("Asin") { op, grad ->
    /**Returns grad * 1/sqrt(1-x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf.square(x)
      val one = tf.const(grad.dataType, 1)
      val den = tf.sqrt(tf.sub(one, x2))
      val inv = tf.reciprocal(den)
      listOf(grad * inv)
    }
  }
  register("Acos") { op, grad ->
    /**Returns grad * -1/sqrt(1-x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf.square(x)
      val one = tf.const(grad.dataType, 1)
      val den = tf.sqrt(tf.sub(one, x2))
      val inv = tf.reciprocal(den)
      listOf(-grad * inv)
    }
  }
  register("Atan") { op, grad ->
    /**Returns grad * 1/ (1 + x^2).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      x = tf.conj(x)
      val x2 = tf.square(x)
      val one = tf.const(grad.dataType, 1)
      val inv = tf.reciprocal(tf.add(one, x2))
      listOf(grad * inv)
    }
  }
  register("Atan2") { op, grad ->
    /**Returns grad * x / (x^2 + y^2), grad * -y / (x^2 + y^2).*/
    val grad = grad[0]!!.toOutput()
    val y = op.inputs[0]
    val x = op.inputs[1]
    return@register tf.controlDependencies(grad) {
      val gradInv = grad / listOf(tf.square(x) + tf.square(y))
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
        xShape.isFullyDefined
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
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    return@register listOf(tf.reshape(tf.sum(grad, rx), sx), tf.reshape(tf.sum(grad, ry), sy))
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
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    return@register listOf(tf.reshape(tf.sum(grad, rx), sx),
                           tf.reshape(-tf.sum(grad, ry), sy))
  }
  register("Mul") { op, grad ->
    /**The gradient of scalar multiplication.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    if (shapesFullySpecifiedAndEqual(x, y, grad) && (grad.dataType == INT32 || grad.dataType == FLOAT)) {
      return@register listOf(tf.mul(grad, y), tf.mul(grad, x))
    }
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = gen_array_ops.broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    return@register listOf(tf.reshape(tf.sum(tf.mul(grad, y), rx), sx),
                           tf.reshape(tf.sum(tf.mul(x, grad), ry), sy))
  }
  register("Div") { op, grad ->
    /**The gradient for the Div operator.*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    return@register listOf(tf.reshape(tf.sum(tf.div(grad, y), rx), sx), tf.reshape(tf.sum(grad * tf.div(tf.div(-x, y), y), ry), sy))
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
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    val floorXy = tf.floorDiv(x, y)
    val gx = tf.reshape(tf.sum(grad, rx), sx)
    val gy = tf.reshape(tf.sum(grad * tf.neg(floorXy), ry), sy)
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
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    return@register listOf(tf.reshape(tf.sum(tf.realDiv(grad, y), rx), sx),
                           tf.reshape(tf.sum(grad * tf.realDiv(tf.realDiv(-x, y), y), ry), sy))
  }
  register("Pow") { op, grad ->
    /**Returns grad * (y*x^(y-1), z*log(x)).*/
    val grad = grad[0]!!.toOutput()
    var x = op.inputs[0]
    var y = op.inputs[1]
    var z = op.outputs[0]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    x = tf.conj(x)
    y = tf.conj(y)
    z = tf.conj(z)
    val gx = tf.reshape(tf.sum(grad * y * tf.pow(x, y - 1), rx), sx)
    val logX = if (x.dataType.isComplex)
      tf.where(tf.notEqual(x, tf.const(0)), tf.log(x), tf.zerosLike(x))
    else
      tf.where(tf.greater(x, tf.const(0)), tf.log(x), tf.zerosLike(x))
    val gy = tf.reshape(tf.sum(grad * z * logX, ry), sy)
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
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    val xgrad = tf.where(xmask, grad, zeros)
    val ygrad = tf.where(xmask, zeros, grad)
    val gx = tf.reshape(tf.sum(xgrad, rx), sx)
    val gy = tf.reshape(tf.sum(ygrad, ry), sy)
    return listOf(gx, gy)
  }
  register("Maximum") { op, grad ->
    /**Returns grad*(x > y, x <= y) with type of grad.*/
    return@register maximumMinimumGrad(op, grad) { x, y -> tf.greaterEqual(x, y) }
  }
  register("Minimum") { op, grad ->
    /**Returns grad*(x < y, x >= y) with type of grad.*/
    return@register maximumMinimumGrad(op, grad) { x, y -> tf.lessEqual(x, y) }
  }
  register("SquaredDifference") { op, grad ->
    /**Returns the gradient for (x-y)^2.*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val y = op.inputs[1]
    val sx = tf.shape(x)
    val sy = tf.shape(y)
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    val xGrad = tf.controlDependencies(grad) {
      tf.scalarMul(tf.const(grad.dataType, 2.0), grad) * (x - y)
    }
    return@register listOf(tf.reshape(tf.sum(xGrad, rx), sx),
                           -tf.reshape(tf.sum(xGrad, ry), sy))
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
        gradA = tf.matMul(grad, b, transposeB = true)
        gradB = tf.matMul(a, grad, transposeA = true)
      }
      !tA && tB -> {
        gradA = tf.matMul(grad, b)
        gradB = tf.matMul(grad, a, transposeA = true)
      }
      tA && !tB -> {
        gradA = tf.matMul(b, grad, transposeB = true)
        gradB = tf.matMul(a, grad)
      }
      tA && tB -> {
        gradA = tf.matMul(b, grad, transposeA = true, transposeB = true)
        gradB = tf.matMul(grad, a, transposeA = true, transposeB = true)
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
        prod = tf.cast(prod, outDtype)
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
    val (rx, ry) = tf.broadcastGradientArgs(sx, sy)
    return@register listOf(tf.reshape(tf.sum(tf.real(grad), rx), sx), tf.reshape(tf.sum(tf.imag(grad), ry), sy))
  }
  register("Real") { op, grad ->
    /**Returns 'grad' as the real part and set the imaginary part 0.*/
    val grad = grad[0]!!.toOutput()
    val zero = tf.const(grad.dataType, 0)
    return@register listOf(tf.complex(grad, zero))
  }
  register("Imag") { op, grad ->
    /**Returns 'grad' as the imaginary part and set the real part 0.*/
    val grad = grad[0]!!.toOutput()
    val zero = tf.const(grad.dataType, 0)
    return@register listOf(tf.complex(zero, grad))
  }
  register("Angle") { op, grad ->
    /**Returns -grad / (Im(x) + iRe(x))*/
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    return@register tf.controlDependencies(grad) {
      val re = tf.real(x)
      val im = tf.imag(x)
      val z = tf.reciprocal(tf.complex(im, re))
      val zero = tf.const(grad.dataType, 0)
      val complexGrad = tf.complex(grad, zero)
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
    return@register listOf(tf.complex(grad, tf.zerosLike(grad)) * tf.sign(op.inputs[0]))
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
    return@register listOf(tf.cross(v, grad), tf.cross(grad, u))
  }
  register("Cumsum") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val axis = op.inputs[1]
    val exclusive = op.attrBool("exclusive")
    val reverse = op.attrBool("reverse")
    return@register listOf(tf.cumsum(grad, axis, exclusive = exclusive, reverse = !reverse), null)
  }
  register("Cumprod") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val x = op.inputs[0]
    val axis = op.inputs[1]
    val exclusive = op.attrBool("exclusive")
    val reverse = op.attrBool("reverse")
    val prod = tf.cumprod(x, axis, exclusive = exclusive, reverse = reverse)
    val out = tf.cumsum(prod * grad, axis, exclusive = exclusive, reverse = !reverse)
    return@register listOf(out / x, null)
  }
}