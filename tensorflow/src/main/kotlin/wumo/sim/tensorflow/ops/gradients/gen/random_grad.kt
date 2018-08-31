import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.get
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.ops.times
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.INT32

fun register_random_grad() {
  /**Gradients for operators defined in random_ops.py.*/
/* from__future__importabsolute_import */
/* from__future__importdivision */
/* from__future__importprint_function */
/* fromtensorflow.python.frameworkimportdtypes */
/* fromtensorflow.python.frameworkimportops */
/* fromtensorflow.python.opsimportarray_ops */
/* fromtensorflow.python.opsimportgen_random_ops */
/* fromtensorflow.python.opsimportmath_ops */
  fun addLeadingUnitDimensions(x: Output, numDimensions: Output): Output {
    val newShape = tf.concatV2(listOf(tf.ones(numDimensions, dtype = INT32),
                                      tf.shape(x)), axis = tf.const(0))
    return tf.reshape(x, newShape)
  }
  register("RandomGamma") { op, grad ->
    val grad = grad[0]!!.toOutput()
    /**Returns the gradient of a Gamma sample w.r.t. alpha.
    
    The gradient is computed using implicit differentiation, see
    "Implicit Reparameterization Gradients" (https://arxiv.org/abs/1805.08498).
    
    Args:
    op: A `RandomGamma` operation. We assume that the inputs to the operation
    are `shape` and `alpha` tensors, and the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
    `op.outputs[0]`.
    
    Returns:
    A `Tensor` with derivatives `dloss / dalpha`
     */
    val shape = op.inputs[0]
    val alpha = op.inputs[1]
    val sample = op.outputs[0]
    tf.controlDependencies(grad) {
      val numSampleDimensions = tf.shape(shape)[0]
      val alphaBroadcastable = addLeadingUnitDimensions(alpha, numSampleDimensions)
      val partialA = tf.randomGammaGrad(alphaBroadcastable, sample)
      listOf(null, tf.sum(grad * partialA,
                          axis = tf.range(tf.const(0), numSampleDimensions)))  //return@register
    }
  }
}