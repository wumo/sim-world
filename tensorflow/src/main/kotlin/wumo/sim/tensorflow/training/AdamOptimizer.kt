package wumo.sim.tensorflow.training
//
//import wumo.sim.tensorflow.*
//import wumo.sim.tensorflow.ops.*
//import wumo.sim.tensorflow.ops.control_flow_ops.group
//import wumo.sim.tensorflow.ops.gen.applyAdam
//import wumo.sim.tensorflow.ops.variables.Variable
//import wumo.sim.util.tuple2
//
///**
// * Construct a new Adam optimizer.
//
//Initialization:
//
//```
//m_0 <- 0 (Initialize initial 1st moment vector)
//v_0 <- 0 (Initialize initial 2nd moment vector)
//t <- 0 (Initialize timestep)
//```
//
//The update rule for `variable` with gradient `g` uses an optimization
//described at the end of section2 of the paper:
//
//```
//t <- t + 1
//lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
//
//m_t <- beta1 * m_{t-1} + (1 - beta1) * g
//v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
//variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
//```
//
//The default value of 1e-8 for epsilon might not be a good default in
//general. For example, when training an Inception network on ImageNet a
//current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
//formulation just before Section 2.1 of the Kingma and Ba paper rather than
//the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
//hat" in the paper.
//
//The sparse implementation of this algorithm (used when the gradient is an
//IndexedSlices object, typically because of `tf.gather` or an embedding
//lookup in the forward pass) does apply momentum to variable slices even if
//they were not used in the forward pass (meaning they have a gradient equal
//to zero). Momentum decay (beta1) is also applied to the entire momentum
//accumulator. This means that the sparse behavior is equivalent to the dense
//behavior (in contrast to some momentum implementations which ignore momentum
//unless a variable slice was actually used).
//
//
// * @param learning_rate: A Output or a floating point value.  The learning rate.
// * @param beta1: A float value or a constant float tensor.
//The exponential decay rate for the 1st moment estimates.
// * @param beta2: A float value or a constant float tensor.
//The exponential decay rate for the 2nd moment estimates.
// * @param epsilon: A small constant for numerical stability. This epsilon is
//"epsilon hat" in the Kingma and Ba paper (in the formula just before
//Section 2.1), not the epsilon in Algorithm 1 of the paper.
// * @param use_locking: If True use locks for update operations.
// * @param name: Optional name for the operations created when applying gradients.
//Defaults to "Adam".
// */
//class AdamOptimizer(val learningRate: Float = 0.001f,
//                    val beta1: Float = 0.9f,
//                    val beta2: Float = 0.999f,
//                    val epsilon: Float = 1e-8f,
//                    use_locking: Boolean = false,
//                    name: String = "Adam") : Optimizer(use_locking, name) {
//  lateinit var lr_t: Output
//  lateinit var beta1_t: Output
//  lateinit var beta2_t: Output
//  lateinit var epsilon_t: Output
//
//  override fun create_slots(var_list: List<Variable>) {
//    // Create the beta1 and beta2 accumulators on the same device as the first
//    // variable. Sort the var_list to make sure this device is consistent across
//    // workers (these need to go on the same PS, otherwise some updates are
//    // silently ignored).
//    val first_var = var_list.minBy { it.name }!!
//    create_non_slot_variable(initial_value = beta1,
//                             name = "beta1_power",
//                             colocate_with = first_var)
//    create_non_slot_variable(initial_value = beta2,
//                             name = "beta2_power",
//                             colocate_with = first_var)
//    // Create slots for the first and second moments.
//    for (v in var_list) {
//      zero_slot(v, "m", name)
//      zero_slot(v, "v", name)
//    }
//  }
//
//  override fun prepare() {
//    lr_t = tf.const(learningRate, name = "learning_rate")
//    beta1_t = tf.const(beta1, name = "beta1")
//    beta2_t = tf.const(beta2, name = "beta2")
//    epsilon_t = tf.const(epsilon, name = "epsilon")
//  }
//
//  override fun apply_dense(grad: Output, _v: Variable): Op {
//    val m = get_slot(_v, "m")
//    val v = get_slot(_v, "v")
//    val (beta1_power, beta2_power) = get_beta_accumulators()
//    return tf.applyAdam(_v, m, v,
//                         tf.cast(beta1_power, _v.dtype.base_dtype),
//                         tf.cast(beta2_power, _v.dtype.base_dtype),
//                         tf.cast(lr_t, _v.dtype.base_dtype),
//                         tf.cast(beta1_t, _v.dtype.base_dtype),
//                         tf.cast(beta2_t, _v.dtype.base_dtype),
//                         tf.cast(epsilon_t, _v.dtype.base_dtype),
//                         grad, use_locking = use_locking).op!!
//  }
//
//  override fun finish(update_ops: MutableList<Op>, name: String): Op {
//    tf.control_dependencies(update_ops) {
//      val (beta1_power, beta2_power) = get_beta_accumulators()
//      tf.colocate_with(beta1_power) {
//        val update_beta1 = beta1_power.assign(beta1_power * beta1_t, use_locking = use_locking).op
//        val update_beta2 = beta2_power.assign(beta2_power * beta2_t, use_locking = use_locking).op
//        update_ops += update_beta1!!
//        update_ops += update_beta2!!
//        return tf.group(update_ops, name)
//      }
//    }
//  }
//
//  private fun get_beta_accumulators() =
//      tuple2(get_non_slot_variable("beta1_power"),
//             get_non_slot_variable("beta2_power"))
//
//}