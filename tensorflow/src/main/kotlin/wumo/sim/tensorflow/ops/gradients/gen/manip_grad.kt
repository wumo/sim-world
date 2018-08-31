import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.ops.unaryMinus
import wumo.sim.tensorflow.tf

fun register_manip_grad() {
  /**Gradients for operators defined in manip_ops.py.*/
/* from__future__importabsolute_import */
/* from__future__importdivision */
/* from__future__importprint_function */
/* fromtensorflow.python.frameworkimportops */
/* fromtensorflow.python.opsimportmanip_ops */
  register("Roll") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val shift = op.inputs[1]
    val axis = op.inputs[2]
    val rollGrad = tf.roll(grad, -shift, axis)
    listOf(rollGrad, null, null)  //return@register
  }
}