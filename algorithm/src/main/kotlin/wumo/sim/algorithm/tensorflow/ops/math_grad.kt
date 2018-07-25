package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.gradients.noGradient
import wumo.sim.algorithm.tensorflow.ops.gradients.register_gradient_op
import wumo.sim.algorithm.tensorflow.ops.gradients.register_no_gradient_op
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.a
import wumo.sim.util.i

fun conjugateHelper(out: Tensor) =
    if (out.dtype == DT_COMPLEX64 || out.dtype == DT_COMPLEX128)
      tf.conj(out)
    else
      out

fun register_math_grad() {
  register_no_gradient_op("Less",
                          "LessEqual",
                          "Greater",
                          "GreaterEqual",
                          "Equal",
                          "ApproximateEqual",
                          "NotEqual",
                          "LogicalAnd",
                          "LogicalOr",
                          "LogicalNot",
                          "Floor")
  
  register_gradient_op("Abs") { op, grad_inputs, grad_outputs ->
    // dx = dy * sign(x)
    grad_outputs.add(tf.mul(grad_inputs[0], tf.sign(op.inputs[0])))
  }
  
  register_gradient_op("Neg") { op, grad_inputs, grad_outputs ->
    // dx = -dy;
    grad_outputs.add(tf.neg(grad_inputs[0]))
  }
  
  register_gradient_op("Inv", "Reciprocal") { op, grad_inputs, grad_outputs ->
    // Use the built-in operator.
    grad_outputs.add(tf.reciprocalGrad(op.outputs[0], grad_inputs[0]))
  }
  
  register_gradient_op("Square") { op, grad_inputs, grad_outputs ->
    // dy/dx = (2 * x)
    val two = tf.cast(tf.const(2), op.inputs[0].dtype)
    val dydx = tf.mul(two, op.inputs[0])
    // grad(x) = grad(y) * conj(dy/dx)
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Sqrt") { op, grad_inputs, grad_outputs ->
    // Use the built-in operator.
    grad_outputs.add(tf.sqrtGrad(op.outputs[0], grad_inputs[0]))
  }
  
  register_gradient_op("Rsqrt") { op, grad_inputs, grad_outputs ->
    // Use the built-in operator.
    grad_outputs.add(tf.rsqrtGrad(op.outputs[0], grad_inputs[0]))
  }
  
  register_gradient_op("Exp") { op, grad_inputs, grad_outputs ->
    // dy/dx = exp(x) = y
    // grad(x) = grad(y) * conj(dy/dx)
    //         = grad(y) * conj(y)
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(op.outputs[0])))
  }
  
  register_gradient_op("Expm1") { op, grad_inputs, grad_outputs ->
    // y = expm1(x)
    // dy/dx = exp(x)
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.exp(op.inputs[0])
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Log") { op, grad_inputs, grad_outputs ->
    // y = log(x)
    // dy/dx = 1 / x
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.reciprocal(op.inputs[0])
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Log1p") { op, grad_inputs, grad_outputs ->
    // y = log1p(x)
    // dy/dx = 1 / (1 + x)
    // grad(x) = grad(y) * conj(dy/dx)
    val one = tf.cast(tf.const(1.0), op.inputs[0].dtype)
    val dydx = tf.reciprocal(tf.add(one, op.inputs[0]))
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Sinh") { op, grad_inputs, grad_outputs ->
    // y = sinh(x)
    // dy/dx = cosh(x)
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.cosh(op.inputs[0])
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  register_gradient_op("Cosh") { op, grad_inputs, grad_outputs ->
    // y = cosh(x)
    // dy/dx = sinh(x)
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.sinh(op.inputs[0])
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Tanh") { op, grad_inputs, grad_outputs ->
    // Use the built-in operator.
    // Note that the built-in operator does not return the conjugate of
    // the gradient.
    val grad = grad_inputs[0]
    // Optimization to avoid calculating conj(y) until the gradient is
    // evaluated.
    tf.control_dependencies(Operation(tf.g, tensorflow.TF_Operation(grad.node()))) {
      val y = conjugateHelper(op.outputs[0])
      grad_outputs.add(tf.tanhGrad(y, grad))
    }
  }
  
  register_gradient_op("Asinh") { op, grad_inputs, grad_outputs ->
    // y = asinh(x)
    // dy/dx = 1 / cosh(y)
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.reciprocal(tf.cosh(op.outputs[0]))
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Acosh") { op, grad_inputs, grad_outputs ->
    // y = acosh(x)
    // dy/dx = 1 / sinh(y)
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.reciprocal(tf.sinh(op.outputs[0]))
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  register_gradient_op("Atanh") { op, grad_inputs, grad_outputs ->
    // y = atanh(x)
    // dy/dx = 1 / (1 - x^2)
    // grad(x) = grad(y) * conj(dy/dx)
    val one = tf.cast(tf.const(1.0), op.inputs[0].dtype)
    val dydx = tf.reciprocal(tf.sub(one, tf.square(op.inputs[0])))
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Sigmoid") { op, grad_inputs, grad_outputs ->
    // Use the built-in operator.
    // Note that the built-in operator does not return the conjugate of
    // the gradient.
    // Optimization to avoid calculating conj(y) until the gradient is
    // evaluated.
    val grad = grad_inputs[0]
    tf.control_dependencies(grad.op!!) {
      val y = conjugateHelper(op.outputs[0])
      grad_outputs.add(tf.sigmoidGrad(y, grad))
    }
  }
  
  register_gradient_op("Sign") { op, grad_inputs, grad_outputs ->
    val shape = tf.shape(op.inputs[0])
    val zero = tf.cast(tf.const(0.0), op.inputs[0].dtype)
    val dx = tf.fill(shape, zero)
    grad_outputs.add(dx)
  }
  
  register_gradient_op("Sin") { op, grad_inputs, grad_outputs ->
    // y = sin(x)
    // dy/dx = cos(x)
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.cos(op.inputs[0])
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  register_gradient_op("Cos") { op, grad_inputs, grad_outputs ->
    // y = cos(x)
    // dy/dx = -sin(x)
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.neg(tf.sin(op.inputs[0]))
    grad_outputs.add(tf.mul(grad_inputs[0], conjugateHelper(dydx)))
  }
  
  register_gradient_op("Asin") { op, grad_inputs, grad_outputs ->
    // y = asin(x)
    // dy/dx = 1 / sqrt(1 - x^2)
    // grad(x) = grad(y) * conj(dy/dx)
    val x2 = tf.square(op.inputs[0])
    val one = tf.cast(tf.const(1.0), op.inputs[0].dtype)
    val dydx = tf.reciprocal(tf.sqrt(tf.sub(one, x2)))
    val dx = tf.mul(grad_inputs[0], conjugateHelper(dydx))
    grad_outputs.add(dx)
  }
  
  register_gradient_op("Acos") { op, grad_inputs, grad_outputs ->
    // y = acos(x)
    // dy/dx = - 1 / (1 - x * x)^1/2
    // dx = dy * (- 1 / (1 - x * x)^1/2)
    val x2 = tf.square(op.inputs[0])
    val one = tf.cast(tf.const(1.0), op.inputs[0].dtype)
    val dydx = tf.neg(tf.reciprocal(tf.sqrt(tf.sub(one, x2))))
    val dx = tf.mul(grad_inputs[0], dydx)
    grad_outputs.add(dx)
  }
  
  register_gradient_op("Tan") { op, grad_inputs, grad_outputs ->
    // y = tan(x)
    // dy/dx = sec(x)^2 = 1 / cos(x)^2
    // grad(x) = grad(y) * conj(dy/dx)
    val dydx = tf.square(tf.reciprocal(tf.cos(op.inputs[0])))
    val dx = tf.mul(grad_inputs[0], conjugateHelper(dydx))
    grad_outputs.add(dx)
  }
  
  register_gradient_op("Atan") { op, grad_inputs, grad_outputs ->
    // y = arctan(x)
    // dy/dx = 1 / (1 + x^2)
    // dx = dy * (1 / (1 + x^2)
    val one = tf.cast(tf.const(1.0), op.inputs[0].dtype)
    val dydx = tf.reciprocal(tf.add(one, tf.square(op.inputs[0])))
    val dx = tf.mul(grad_inputs[0], dydx)
    grad_outputs.add(dx)
  }
  
  fun binaryGradCommon(op: Operation, grad_outputs: MutableList<Tensor>, gx_1: Tensor, gx_2: Tensor) {
    val sx_1 = tf.shape(op.inputs[0])
    val sx_2 = tf.shape(op.inputs[1])
    val (r0, r1) = tf.broadcastGradientArgs(sx_1, sx_2)
    val dx_1 = tf.reshape(tf.sum(gx_1, r0), sx_1)
    val dx_2 = tf.reshape(tf.sum(gx_2, r1), sx_2)
    grad_outputs.add(dx_1)
    grad_outputs.add(dx_2)
  }
  
  register_gradient_op("Add") { op, grad_inputs, grad_outputs ->
    // y = x_1 + x_2
    // dy/dx_1 = dy/dx_2 = 1
    val gx_1 = tf.identity(grad_inputs[0])
    val gx_2 = tf.identity(grad_inputs[0])
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  
  register_gradient_op("Sub") { op, grad_inputs, grad_outputs ->
    // y = x_1 - x_2
    // dy/dx_1 = 1
    // dy/dx_2 = -1
    val gx_1 = tf.identity(grad_inputs[0])
    val gx_2 = tf.neg(grad_inputs[0])
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  
  register_gradient_op("Mul") { op, grad_inputs, grad_outputs ->
    // y = x_1 * x_2
    // dy/dx_1 = x_2
    // dy/dx_2 = x_1
    val x_1 = conjugateHelper(op.inputs[0])
    val x_2 = conjugateHelper(op.inputs[1])
    val gx_1 = tf.mul(grad_inputs[0], x_2)
    val gx_2 = tf.mul(grad_inputs[0], x_1)
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  register_gradient_op("Div") { op, grad_inputs, grad_outputs ->
    // y = x_1 / x_2
    // dy/dx_1 = 1/x_2
    // dy/dx_2 = -x_1/x_2^2
    val x_1 = conjugateHelper(op.inputs[0])
    val x_2 = conjugateHelper(op.inputs[1])
    val gx_1 = tf.div(grad_inputs[0], x_2)
    val gx_2 = tf.mul(grad_inputs[0],
                      tf.div(tf.div(tf.neg(x_1), x_2), x_2))
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  
  register_gradient_op("RealDiv") { op, grad_inputs, grad_outputs ->
    // y = x_1 / x_2
    // dy/dx_1 = 1/x_2
    // dy/dx_2 = -x_1/x_2^2
    val x_1 = conjugateHelper(op.inputs[0])
    val x_2 = conjugateHelper(op.inputs[1])
    val gx_1 = tf.realDiv(grad_inputs[0], x_2)
    val gx_2 = tf.mul(grad_inputs[0],
                      tf.realDiv(tf.realDiv(tf.neg(x_1), x_2), x_2))
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  
  register_gradient_op("SquaredDifference") { op, grad_inputs, grad_outputs ->
    // y = (x_1 - x_2)^2
    // dy/dx_1 = 2 * (x_1 - x_2)
    // dy/dx_2 = -2 * (x_1 - x_2)
    val x_1 = conjugateHelper(op.inputs[0])
    val x_2 = conjugateHelper(op.inputs[1])
    val two = tf.cast(tf.const(2), grad_inputs[0].dtype)
    val gx_1 = tf.mul(grad_inputs[0], tf.mul(two, tf.sub(x_1, x_2)))
    val gx_2 = tf.neg(gx_1)
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  
  register_gradient_op("AddN") { op, grad_inputs, grad_outputs ->
    // AddN doesn't support broadcasting, so all the inputs must be the
    // same shape.
    // Note:
    // dy/dx_k = d(x_1 + x_2 + ... + x_n)/dx_k = 1 for all x_k
    // hence dx_k = dy for all x_k
    // So the gradient for AddN just transfers the incoming gradient to
    // all outgoing gradients.
    val incoming = tf.identity(grad_inputs[0])
    for (i in 0 until op.inputs.size)
      grad_outputs.add(incoming)
  }
  register_gradient_op("Pow") { op, grad_inputs, grad_outputs ->
    val x = conjugateHelper(op.inputs[0])
    val y = conjugateHelper(op.inputs[1])
    val z = conjugateHelper(op.outputs[0])
    val grad = grad_inputs[0]
    // grad * y * pow(x, y - 1)
    val one = tf.cast(tf.const(1.0), y.dtype)
    val gx_1 = tf.mul(tf.mul(grad, y),
                      tf.pow(x, tf.sub(y, one)))
    // Avoid false singularity at x = 0
    val x_dtype = x.dtype
    val zero = tf.cast(tf.const(0.0), x_dtype)
    if (x_dtype == DT_COMPLEX64 || x_dtype == DT_COMPLEX128) {
      // real(x) < 0 is fine for the complex case
      val log_x = tf.where(tf.notEqual(x, zero), tf.log(x), tf.zerosLike(x))
      val gy_1 = tf.mul(tf.mul(grad, z), log_x)
      binaryGradCommon(op, grad_outputs, gx_1, gy_1)
    } else {
      // There's no sensible real value to return if x < 0, so return 0
      val log_x = tf.where(tf.greater(x, zero), tf.log(x), tf.zerosLike(x))
      val gy_1 = tf.mul(tf.mul(grad, z), log_x)
      binaryGradCommon(op, grad_outputs, gx_1, gy_1)
    }
  }
  /**
   * MaximumMinimumGradCommon adds shared ops to calculate gradients for
   * the binary Maximum and Minimum ops.
   */
  fun maximumMinimumGradCommon(op: Operation,
                               grad_inputs: List<Tensor>,
                               grad_outputs: MutableList<Tensor>,
                               comparator: Tensor) {
    // comparator is a boolean tensor, with
    // y = x_1 at points where comparator is true, and x_2 otherwise
    // Therefore
    // dy/dx_1 = 1 where comparator is true, and 0 otherwise.
    // dy/dx_2 = 0 where comparator is true, and 1 otherwise.
    val grad = grad_inputs[0]
    val zeros = tf.zerosLike(grad)
    val gx_1 = tf.where(comparator, grad, zeros)
    val gx_2 = tf.where(comparator, zeros, grad)
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  
  register_gradient_op("Maximum") { op, grad_inputs, grad_outputs ->
    val comparator = tf.greaterEqual(op.inputs[0], op.inputs[1])
    maximumMinimumGradCommon(op, grad_inputs, grad_outputs, comparator)
  }
  
  register_gradient_op("Minimum") { op, grad_inputs, grad_outputs ->
    val comparator = tf.lessEqual(op.inputs[0], op.inputs[1])
    maximumMinimumGradCommon(op, grad_inputs, grad_outputs, comparator)
  }
  
  register_gradient_op("Real") { op, grad_inputs, grad_outputs ->
    val zero = tf.cast(tf.const(0.0), op.outputs[0].dtype)
    val dx = tf.complex(grad_inputs[0], zero)
    grad_outputs.add(dx)
  }
  
  register_gradient_op("Imag") { op, grad_inputs, grad_outputs ->
    val zero = tf.cast(tf.const(0.0), op.outputs[0].dtype)
    val dx = tf.complex(zero, grad_inputs[0])
    grad_outputs.add(dx)
  }
  
  register_gradient_op("Complex") { op, grad_inputs, grad_outputs ->
    val gx_1 = tf.real(grad_inputs[0])
    val gx_2 = tf.imag(grad_inputs[0])
    binaryGradCommon(op, grad_outputs, gx_1, gx_2)
  }
  
  register_gradient_op("Angle") { op, grad_inputs, grad_outputs ->
    // y = Angle(x)
    // dx = -dy / (Im(x) + iRe(x)) = -dy * z
    val re = tf.real(op.inputs[0])
    val im = tf.imag(op.inputs[0])
    val z_inv = tf.reciprocal(tf.complex(im, re))
    val zero = tf.cast(tf.const(0), grad_inputs[0].dtype)
    val grad = tf.complex(grad_inputs[0], zero)
    val dx = tf.neg(tf.mul(grad, z_inv))
    grad_outputs.add(dx)
  }
  
  register_gradient_op("Conj") { op, grad_inputs, grad_outputs ->
    grad_outputs.add(tf.conj(grad_inputs[0]))
  }
  /**
   * Helper function for reduction ops.
   *
   * input_shape: 1-D Tensor, the shape of the Tensor being reduced.
   * axes: 1-D Tensor, the reduction axes.
   *   Note that the reduction indices are in the range
   *   -rank(input_shape), rank(input_shape)
   * returns a 1-D Tensor, the output shape as if keep_dims were set to True.
   */
  fun reducedShapeHelper(input_shape: Tensor, reduction_axes: Tensor): Tensor {
    val zero = tf.const(0)
    val one = tf.const(1)
    
    // Running example in comments
    // input_shape = [2, 3, 5, 7]
    // axes = [1, 2]
    // The result (a shape after a reduction with keep_dims=True)
    // [2, 1, 1, 7]
    //
    // We can treat each entry in axes as an index into input_shape that
    // should be replaced by 1.
    // We use DynamicStitch to do this.
    
    // input_rank = 4
    val input_rank = tf.size(input_shape)
    
    // Normalize any negative indices in the reduction_axes to positive
    // values.
    val axes = tf.mod(tf.add(reduction_axes, input_rank), input_rank)
    
    // This [0..input_rank) range of integers is used in DynamicStitch to
    // first copy input_shape to the result.
    // input_rank_range = [0, 1, 2, 3]
    val input_rank_range = tf.range(zero, input_rank, one)
    
    // A 1-filled tensor with the same shape as axes. DynamicStitch will
    // merge these 1s (using axes for indices) to the correct
    // position in the result.
    // axes_ones = [1, 1]
    val axes_ones = tf.onesLike(axes)
    
    // using DynamicStitch:
    // indices = { input_rank_range, axes }
    //         = { [0, 1, 2, 3], [1, 2] }
    // data = { input_shape, axes_ones }
    //      = { [2, 3, 5, 7], [1, 1] }
    // The input_rank_range entry in indices first replicates the
    // input_shape to the result.
    // The axes entry in indices then moves a 1 to each of its entries,
    // resulting in
    // [2, 1, 1, 7]
    val indices = a(input_rank_range, axes)
    val data = a(input_shape, axes_ones)
    return tf.dynamicStitch(indices, data)
  }
  
  /**
   * Integer division x / y, assuming x and y >=0, but treats x/0 = x
   */
  fun safeDivHelper(x: Tensor, y: Tensor) =
      tf.div(x, tf.maximum(y, tf.const(1)))
  
  /**
   * SumGradHelper returns the gradient for the Sum operator, and is used
   * by SumGrad and MeanGrad.
   */
  fun sumGradHelper(op: Operation, grad_inputs: List<Tensor>): Tensor {
    // The partial derivative for any input along a "reduced" dimension
    // is just 1, so we only need replicate the output gradient on such a
    // dimension to its "expanded" shape.
    // Running example:
    // input is
    // [[a, b, c],
    //  [d, e, f]]
    // reduction_indices = [1]
    // Sum = [a + b + c, d + e + f]
    // if the gradient is [g1, g2]
    // We want the propagated gradient to be
    // [[g1, g1, g1],
    //  [g2, g2, g2]]
    
    // input_shape = [2, 3]
    val input_shape = tf.shape(op.inputs[0])
    
    // output_shape_kept_dims = [2, 1]
    val output_shape_kept_dims = reducedShapeHelper(input_shape, op.inputs[1])
    
    // This step "flips" any 1s with values from the input_shape, and
    // replaces remaining entries with 1. This creates a shape that
    // shows how much each dimension in the incoming gradient should be
    // replicated.
    // tile_scaling = [1, 3]
    val title_scaling = safeDivHelper(input_shape, output_shape_kept_dims)
    
    // grad = [[g1], [g2]]
    val grad = tf.reshape(grad_inputs[0], output_shape_kept_dims)
    
    // tile(grad, tile_scaling) = [[g1, g1, g1], [g2, g2, g2]]
    return tf.tile(grad, title_scaling)
  }
  
  register_gradient_op("Sum") { op, grad_inputs, grad_outputs ->
    grad_outputs.add(sumGradHelper(op, grad_inputs))
    grad_outputs.add(noGradient)
  }
  
  register_gradient_op("Mean") { op, grad_inputs, grad_outputs ->
    // The Mean gradient is just like the Sum gradient, except that
    // all gradients are also divided by the size of reduced groups.
    val sum_grad = sumGradHelper(op, grad_inputs)
    
    // The product of all entries in a tensor's shape is the total
    // number of entries in the tensor. This step calculates
    // n_input_entries/n_output_entries
    // = group_size
    val input_shape = tf.shape(op.inputs[0])
    val output_shape = tf.shape(op.outputs[0])
    val zero = tf.const(0)
    val group_size = safeDivHelper(tf.prod(input_shape, zero), tf.prod(output_shape, zero))
    
    // propagate sum_grad/group_size
    grad_outputs.add(tf.div(sum_grad, tf.cast(group_size, sum_grad.dtype)))
    grad_outputs.add(noGradient)
  }
  register_gradient_op("Erf") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val two_over_root_pi = tf.cast(tf.const(2 / Math.sqrt(Math.PI)), grad.dtype)
    tf.control_dependencies(grad.op!!) {
      val x = conjugateHelper(op.inputs[0])
      // grad * 2/sqrt(pi) * exp(-x**2)
      val dx = tf.mul(tf.mul(grad, two_over_root_pi),
                      tf.exp(tf.neg(tf.square(x))))
      grad_outputs.add(dx)
    }
  }
  
  register_gradient_op("Lgamma") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    tf.control_dependencies(grad.op!!) {
      val x = conjugateHelper(op.inputs[0])
      val dx = tf.mul(grad, tf.digamma(x))
      grad_outputs.add(dx)
    }
  }
  
  fun minOrMaxGrad(op: Operation, grad_inputs: List<Tensor>, grad_outputs: MutableList<Tensor>) {
    // The partial derivative for any input along a "reduced" dimension
    // is 1 when it is the min (or max) and 0 everywhere else. So the
    // gradient calculation is identical for both operators.
    //
    // There's a special case for propagating gradients when there are
    // multiple minima (or maxima) - we choose to divide the gradient
    // equally among all matching inputs.
    //
    // Please note this comment
    // https://github.com/tensorflow/tensorflow/issues/4886#issuecomment-256836063
    // for details.
    
    // Running example:
    // input: [[5, 5, 5],
    //         [1, 2, -3]]
    // reduction_indices: [1]
    val input = op.inputs[0]
    val reduction_indices = op.inputs[1]
    
    // [2, 3]
    val input_shape = tf.shape(input)
    
    // [2, 1]
    val output_shape_kept_dims = reducedShapeHelper(input_shape, reduction_indices)
    
    // for op=min (say)
    // output = [5, -3]
    // y = [[5],
    //      [-3]]
    val y = tf.reshape(op.outputs[0], output_shape_kept_dims)
    
    // reshape([g1, g2], [2, 1]) = [[g1],
    //                              [g2]]
    val grad = tf.reshape(grad_inputs[0], output_shape_kept_dims)
    
    // indicators = equal(y, input)
    //  = equal([[5],   [[5, 5, 5],
    //           [-3]],  [1, 2, -3]])
    //  = [[1, 1, 1],
    //     [0, 0, 1]]
    val indicators = tf.cast(tf.equal(y, input), grad_inputs[0].dtype)
    
    // [[3],
    //  [1]]
    val num_selected = tf.reshape(tf.sum(indicators, reduction_indices), output_shape_kept_dims)
    
    // [[1/3, 1/3, 1/3],
    //  [0, 0, 1]]
    val scale = tf.div(indicators, num_selected)
    
    // [[g1/3, g1/3, g1/3],
    //  [0, 0, g2]]
    grad_outputs.add(tf.mul(scale, grad))
    
    // Stop propagation along reduction_indices
    grad_outputs.add(noGradient)
  }
  
  register_gradient_op("Min", "Max") { op, grad_inputs, grad_outputs ->
    minOrMaxGrad(op, grad_inputs, grad_outputs)
  }
  
  register_gradient_op("Prod") { op, grad_inputs, grad_outputs ->
    // The gradient can be expressed by dividing the product by each entry of
    // the input tensor. If our input is
    // [
    //  [3, 4],
    //  [5, 6],
    //  [7, 8]
    // ]
    // and we do a Prod operation on the axis 1, we will obtain [[105, 192]].
    // The gradient will have the same shape as the input
    //     [
    //       [105/3, 192/4],
    // dz *  [105/5, 192/6],
    //       [105/7, 192/6]
    //     ]
    // If the input contains a zero, the division is impossible but
    // if we take the calculation that gave the first gradient
    // (3 * 5 * 6)/3 is equal to 5 * 6
    // the trick will be to cumprod the elements on the axis without
    // the element at the current position (3 in the example above).
    // We will take as example:
    // [
    //   [
    //     [3.0, 4.0],
    //     [5.0, 6.0],
    //     [7.0, 8.0]
    //   ],
    //   [
    //     [3.0, 5.0],
    //     [0.0, 6.0],
    //     [5.0, 6.0]
    //   ]
    // ]
    
    val zero = tf.const(0)
    val one = tf.const(1)
    
    // [2, 3, 2]
    val input_shape = tf.shape(op.inputs[0])
    
    // The Reshape with -1 flattens the reduction indices.
    // [1]
    val reduction_indices = tf.reshape(op.inputs[1], tf.const(i(-1)))
    
    // [2, 1, 2]
    val output_shape_kept_dims = reducedShapeHelper(input_shape, reduction_indices)
    
    // [1, 3, 1]
    val tile_scaling = safeDivHelper(input_shape, output_shape_kept_dims)
    
    // [[[105, 192]], [[0, 180]]]
    val grad = tf.reshape(grad_inputs[0], output_shape_kept_dims)
    
    // [[[105, 192], [105, 192], [105, 192]], [[0, 180], [0, 180], [0, 180]]]
    val grad_tiled = tf.tile(grad, tile_scaling)
    
    tf.begin_device("/cpu:0")
    // [3]
    val rank = tf.rank(op.inputs[0])
    
    // Normalize any negative indices in the reduction_axes to positive values.
    val reduction_indices_pos = tf.mod(tf.add(reduction_indices, rank), rank)
    
    // [1]
    val reduced = tf.cast(reduction_indices_pos, DT_INT32)
    
    // [0, 1, 2]
    val idx = tf.range(zero, rank, one)
    
    // [0, 2]
    val other = tf.listDiff(idx, reduced)[0]
    
    // [1, 0, 2]
    val perm = tf.concat(a(reduced, other), tf.const(0))
    
    // 3 => [3]
    val reduced_num = tf.prod(tf.gather(input_shape, reduced), tf.const(0))
    
    // 2 * 2 => [2]
    val other_num = tf.prod(tf.gather(input_shape, other), tf.const(0))
    tf.end_device()
    // [
    //    [
    //       [ 3.,  4.],
    //       [ 3.,  5.]
    //   ],
    //   [
    //       [ 5.,  6.],
    //       [ 0.,  6.]
    //   ],
    //   [
    //       [ 7.,  8.],
    //       [ 5.,  6.]
    //   ]
    // ]
    val permuted = tf.transpose(op.inputs[0], perm)
    
    // [3, 2, 2]
    val permuted_shape = tf.shape(permuted)
    
    // [
    //   [ 3.,  4.,  3.,  5.],
    //   [ 5.,  6.,  0.,  6.],
    //   [ 7.,  8.,  5.,  6.]
    // ]
    val reshaped = tf.reshape(permuted, tf.stack(a(reduced_num, other_num)))
    
    // [
    //   [ 1.,  1.,  1.,  1.],
    //   [ 3.,  4.,  3.,  5.],
    //   [ 15.,  24.,  0.,  30.]
    // ]
    val left = tf.cumprod(reshaped, zero, exclusive = true)
    
    // [
    //   [ 35.,  48.,  0.,  36.],
    //   [  7.,   8.,   5.,   6.],
    //   [  1.,   1.,   1.,   1.]
    // ]
    val right = tf.cumprod(reshaped, zero, exclusive = true, reverse = true)
    
    // left * right =
    // [
    //   [ 35.,  48.,  0.,  36.],
    //   [ 21.,  32.,  15.,  30.],
    //   [ 15.,  24.,  0.,  30.]
    // ]
    // y =
    // [
    //   [
    //     [ 35.,  48.],
    //     [ 0.,  36.]
    //   ],
    //   [
    //     [ 21.,  32.],
    //     [ 15.,  30.]
    //   ],
    //   [
    //     [ 15.,  24.],
    //     [ 0.,  30.]
    //   ]
    // ]
    val y = tf.reshape(tf.mul(left, right), permuted_shape)
    
    // out =
    // [
    //   [
    //     [ 35.,  48.],
    //     [ 21.,  32.],
    //     [ 15.,  24.]
    //   ],
    //   [
    //     [ 0.,   36.],
    //     [ 15.,  30.],
    //     [ 0.,  30.]
    //   ]
    // ]
    val out = tf.mul(grad_tiled, tf.transpose(y, tf.invertPermutation(perm)))
    
    grad_outputs.add(tf.reshape(out, input_shape))
    
    // stop propagation along reduction_indices
    grad_outputs.add(noGradient)
  }
  
  /**
   * MatMulGrad helper function used to compute two MatMul operations
   * based on input matrix transposition combinations.
   */
  fun matMulGradHelper(is_batch: Boolean,
                       x0: Tensor, adj_x0: Boolean
                       , x1: Tensor, adj_x1: Boolean,
                       y0: Tensor, adj_y0: Boolean,
                       y1: Tensor, adj_y1: Boolean,
                       grad_outputs: MutableList<Tensor>) {
    if (!is_batch) {
      val dx = tf.matmul(x0, x1, transpose_a = adj_x0, transpose_b = adj_x1)
      val dy = tf.matmul(y0, y1, transpose_a = adj_y0, transpose_b = adj_y1)
      grad_outputs.add(dx)
      grad_outputs.add(dy)
    } else {
      val dx = tf.batchMatMul(x0, x1, adj_x = adj_x0, adj_y = adj_x1)
      val dy = tf.batchMatMul(y0, y1, adj_x = adj_y0, adj_y = adj_y1)
      grad_outputs.add(dx)
      grad_outputs.add(dy)
    }
  }
  
  /**
   * MatMulGrad common used to read and check node attr state, and determine
   * proper MatMul products for gradients based on input matrix transposition
   * combinations.
   */
  fun matMulGradCommon(op: Operation,
                       is_batch: Boolean,
                       grad_inputs: List<Tensor>,
                       attr_adj_x: String, attr_adj_y: String,
                       grad_outputs: MutableList<Tensor>) {
    var a = op.inputs[0]
    var b = op.inputs[1]
    // Use conjugate of the inputs for MatMul
    if (!is_batch) {
      a = conjugateHelper(a)
      b = conjugateHelper(b)
    }
    val product = op.outputs[0]
    
    val pa = BoolPointer(1)
    val pb = BoolPointer(1)
    GetNodeAttr(product.node().attrs(), BytePointer(attr_adj_x), pa)
    GetNodeAttr(product.node().attrs(), BytePointer(attr_adj_y), pb)
    val ta = pa.get()
    val tb = pb.get()
    when {
      !ta && !tb ->
        matMulGradHelper(is_batch, grad_inputs[0], false, b, true, a,
                         true, grad_inputs[0], false, grad_outputs)
      !ta && tb ->
        matMulGradHelper(is_batch, grad_inputs[0], false, b, false,
                         grad_inputs[0], true, a, false, grad_outputs)
      ta && !tb ->
        matMulGradHelper(is_batch, b, false, grad_inputs[0], true, a,
                         false, grad_inputs[0], false, grad_outputs)
      else ->
        matMulGradHelper(is_batch, b, true, grad_inputs[0], true,
                         grad_inputs[0], true, a, true, grad_outputs)
    }
  }
  register_gradient_op("MatMul") { op, grad_inputs, grad_outputs ->
    matMulGradCommon(op, false, grad_inputs, "transpose_a", "transpose_b", grad_outputs)
  }
  register_gradient_op("BatchMatMul") { op, grad_inputs, grad_outputs ->
    matMulGradCommon(op, true, grad_inputs, "adj_x", "adj_y", grad_outputs)
  }
}
