package wumo.sim.algorithm.tensorflow.training

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.tuple2

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
  
  override fun create_slots(var_list: List<Variable>) {
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
      zero_slot(v, "m", name)
      zero_slot(v, "v", name)
    }
  }
  
  override fun prepare() {
    lr_t = tf.const(learningRate, name = "learning_rate")
    beta1_t = tf.const(learningRate, name = "beta1")
    beta2_t = tf.const(learningRate, name = "beta2")
    epsilon_t = tf.const(learningRate, name = "epsilon")
  }
  
  override fun apply_dense(grad: Tensor, v: Variable) =
      tf.applyGradientDescent(v, tf.cast(lr_t, v.dtype), grad)
  
  override fun finish(update_ops: MutableList<Operation>, name: String): Operation {
    tf.ctx.control_dependencies(update_ops) {
      val (beta1_power, beta2_power) = get_beta_accumulator()
      tf.ctx.colocate_with(beta1_power) {
        val update_beta1 = beta1_power.assign(beta1_power * beta1_t, use_locking = use_locking)
        val update_beta2 = beta2_power.assign(beta2_power * beta2_t, use_locking = use_locking)
        update_ops += update_beta1
        update_ops += update_beta2
        return tf.group(update_ops, name)
      }
    }
  }
  
  private fun get_beta_accumulator() =
      tuple2(get_non_slot_variable("beta1_power"),
             get_non_slot_variable("beta2_power"))
  
}