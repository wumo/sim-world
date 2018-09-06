import wumo.sim.tensorflow.ops.basic.unaryMinus
import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.tf

fun register_manip_grad() {
  /**Gradients for operators defined in manip_ops.py.*/
  register("Roll") { op, grad ->
    val grad = grad[0]!!.toOutput()
    val shift = op.inputs[1]
    val axis = op.inputs[2]
    val rollGrad = tf.roll(grad, -shift, axis)
    listOf(rollGrad, null, null)  //return@register
  }
}