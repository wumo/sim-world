package wumo.sim.algorithm.tensorflow.training

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.applyGradientDescent
import wumo.sim.algorithm.tensorflow.ops.cast
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.tf

class AdamOptimizer(val learningRate: Float = 0.001f,
                    val beta1: Float = 0.9f,
                    val beta2: Float = 0.999f,
                    val epsilon: Float = 1e-8f,
                    use_locking: Boolean = false,
                    name: String = "Adam") : Optimizer(use_locking, name) {
  lateinit var lr_t: Tensor
  lateinit var beta1_t: Tensor
  lateinit var beta2_t: Tensor
  lateinit var epsilon_t: Tensor
  
  override fun create_slots(var_list: List<Tensor>) {
    // Create the beta1 and beta2 accumulators on the same device as the first
    // variable. Sort the var_list to make sure this device is consistent across
    // workers (these need to go on the same PS, otherwise some updates are
    // silently ignored).
    val first_var = var_list.minBy { it.op.name }!!
    create_non_slot_variable(initial_value = beta1,
                             name = "beta1_power",
                             colocate_with = first_var)
    create_non_slot_variable(initial_value = beta2,
                             name = "beta2_power",
                             colocate_with = first_var)
    // Create slots for the first and second moments.
    for (v in var_list) {
      zero_slot(v,"m",name)
      zero_slot(v,"v",name)
    }
  }
  
  override fun prepare() {
    lr_t = tf.const(learningRate, name = "learning_rate")
    beta1_t = tf.const(learningRate, name = "beta1")
    beta2_t = tf.const(learningRate, name = "beta2")
    epsilon_t = tf.const(learningRate, name = "epsilon")
  }
  
  override fun apply_dense(grad: Tensor, v: Tensor) =
      tf.applyGradientDescent(v, tf.cast(lr_t, v.dtype), grad)
  
}