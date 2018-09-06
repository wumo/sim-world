package wumo.sim.tensorflow.ops.training

import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tensor.Tensor
import wumo.sim.tensorflow.tf

/** Optimizer that implements the Adam optimization algorithm.
 *
 * Initialization:
 * {{{
 *   m_0 = 0  // Initialize the 1st moment vector
 *   v_0 = 0  // Initialize the 2nd moment vector
 *   t = 0    // Initialize the time step
 * }}}
 *
 * The Adam update for step `t` is as follows:
 * {{{
 *   learningRate_t = initialLearningRate * sqrt(beta1 - beta2^t) / (1 - beta1^t)
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * gradient * gradient
 *   variable -= learningRate_t * m_t / (sqrt(v_t) + epsilon)
 * }}}
 *
 * The default value of `1e-8` for epsilon might not be a good default in general. For example, when training an
 * Inception network on ImageNet a current good choice is `1.0` or `0.1`. Note that since the Adam optimizer uses the
 * formulation just before Section 2.1 of the [Kingma and Ba paper](https://arxiv.org/abs/1412.6980) rather than the
 * formulation in Algorithm 1, the "epsilon" referred to here is "epsilon hat" in the paper.
 *
 * The sparse implementation of this algorithm (used when the gradient is an indexed slices object, typically because
 * of `tf.gather` or an embedding lookup in the forward pass) does apply momentum to variable slices even if they were
 * not used in the forward pass (meaning they have a gradient equal to zero). Momentum decay (`beta1`) is also applied
 * to the entire momentum accumulator. This means that the sparse behavior is equivalent to the dense behavior (in
 * contrast to some momentum implementations which ignore momentum unless a variable slice was actually used).
 *
 * For more information on this algorithm, please refer to this [paper](https://arxiv.org/abs/1412.6980)
 * ([PDF](https://arxiv.org/pdf/1412.6980.pdf)).
 *
 * @param  learningRate           Learning rate. Must be `> 0`. If used with `decay`, then this argument
 *                                specifies the initial value of the learning rate.
 * @param  decay                  Learning rate decay method to use for each update.
 * @param  beta1                  Exponential decay rate for the first moment estimates.
 * @param  beta2                  Exponential decay rate for the second moment estimates.
 * @param  useNesterov            If `true`, Nesterov momentum is used for the updates.
 * @param  epsilon                Small constant used for numerical stability. This epsilon corresponds to
 *                                "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1),
 *                                and not to the epsilon in Algorithm 1 of the paper.
 * @param  useLocking             If `true`, the gradient descent updates will be protected by a lock. Otherwise, the
 *                                behavior is undefined, but may exhibit less contention.
 * @param  learningRateSummaryTag Optional summary tag name to use for the learning rate value. If `null`, no summary
 *                                is created for the learning rate. Otherwise, a scalar summary is created which can
 *                                be monitored using TensorBoard.
 * @param  name                   Name for this optimizer.
 *
 * @author Emmanouil Antonios Platanios
 */
open class AdamOptimizer(
    val learningRate: () -> Float = { 0.001f },
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val epsilon: Float = 1e-8f,
    override val useLocking: Boolean = false,
    val learningRateSummaryTag: String? = null,
    override val name: String = "Adam"
) : Optimizer() {
  
  override val ignoreDuplicateSparseIndices = true
  lateinit var learningRateTensor: Output
  lateinit var beta1Tensor: Output
  lateinit var beta2Tensor: Output
  lateinit var epsilonTensor: Output
  
  protected fun getBetaPowerAccumulators(): Pair<Variable, Variable> =
      getNonSlotVariable("beta1_power", tf.currentGraph)!! to
          getNonSlotVariable("beta2_power", tf.currentGraph)!!
  
  override fun createSlots(variables: List<Variable>) {
    val firstVar = variables.minBy { it.name }!!
    getOrCreateNonSlotVariable("beta1_power", Tensor(beta1), mutableSetOf(firstVar.op))
    getOrCreateNonSlotVariable("beta2_power", Tensor(beta2), mutableSetOf(firstVar.op))
    variables.forEach { v ->
      zerosSlot("m", v, name)
      zerosSlot("v", v, name)
    }
  }
  
  override fun prepare(iteration: Variable?) {
    learningRateTensor = tf.const(learningRate(), "learning_rate")
    beta1Tensor = tf.const(beta1, "beta1")
    beta2Tensor = tf.const(beta2, "beta2")
    epsilonTensor = tf.const(epsilon, "epsilon")
  }
  
  override fun applyDense(gradient: Output, variable: Variable, iteration: Variable?): Op {
    val m = getSlot("m", variable)!!
    val v = getSlot("v", variable)!!
    val (beta1Power, beta2Power) = getBetaPowerAccumulators()
    return tf.applyAdam(variable.variable, m.variable, v.variable,
                        tf.cast(beta1Power.value, variable.dataType.baseDataType),
                        tf.cast(beta2Power.value, variable.dataType.baseDataType),
                        tf.cast(learningRateTensor, variable.dataType.baseDataType),
                        tf.cast(beta1Tensor, variable.dataType.baseDataType),
                        tf.cast(beta2Tensor, variable.dataType.baseDataType),
                        tf.cast(epsilonTensor, variable.dataType.baseDataType),
                        gradient,
                        useLocking).op
  }
  
  override fun applySparse(gradient: IndexedSlices, variable: Variable, iteration: Variable?): Op {
    val (beta1Power, beta2Power) = getBetaPowerAccumulators()
    val beta1_power = tf.cast(beta1Power.value, variable.dataType.baseDataType)
    val beta2_power = tf.cast(beta2Power.value, variable.dataType.baseDataType)
    val lr_t = tf.cast(learningRateTensor, variable.dataType.baseDataType)
    val beta1_t = tf.cast(beta1Tensor, variable.dataType.baseDataType)
    val beta2_t = tf.cast(beta2Tensor, variable.dataType.baseDataType)
    val epsilon_t = tf.cast(epsilonTensor, variable.dataType.baseDataType)
    val lr = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))
    // m_t = beta1 * m + (1 - beta1) * g_t
    val m = getSlot("m", variable)!!
    val m_scaled_g_values = gradient.values * (1 - beta1_t)
    var m_t = tf.assign(m.variable, m * beta1_t,
                        useLocking = useLocking)
    tf.controlDependencies(m_t) {
      m_t = tf.scatterAdd(m.variable, gradient.indices, m_scaled_g_values)
    }
    // v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    val v = getSlot("v", variable)!!
    val v_scaled_g_values = (gradient.values * gradient.values) * (1 - beta2_t)
    var v_t = tf.assign(v.variable, v * beta2_t, useLocking = useLocking)
    tf.controlDependencies(v_t) {
      v_t = tf.scatterAdd(v.variable, gradient.indices, v_scaled_g_values)
    }
    val v_sqrt = tf.sqrt(v_t)
    val var_update = tf.assignSub(variable.variable, lr * m_t / (v_sqrt + epsilon_t),
                                  useLocking = useLocking)
    return tf.group(setOf(var_update.op, m_t.op, v_t.op))
  }
  
  override fun finish(updateOps: Set<Op>, nameScope: String): Op {
    val updateBetaOps = tf.controlDependencies(updateOps.toMutableSet()) {
      val (beta1Power, beta2Power) = getBetaPowerAccumulators()
      tf.colocateWith(beta1Power.op) {
        val updateBeta1 = beta1Power.assign(beta1Power * beta1Tensor)
        val updateBeta2 = beta2Power.assign(beta2Power * beta2Tensor)
        setOf(updateBeta1.op, updateBeta2.op)
      }
    }
    return tf.group(updateOps + updateBetaOps, nameScope)
  }
}
//
//import wumo.sim.tensorflow.*
//import wumo.sim.tensorflow.ops.*
//import wumo.sim.tensorflow.ops.control_flow_ops.group
//import wumo.sim.tensorflow.ops.gen.applyAdam
//import wumo.sim.tensorflow.ops.variables.Variable
//import wumo.sim.util.t2
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
//                             colocateWith = first_var)
//    create_non_slot_variable(initial_value = beta2,
//                             name = "beta2_power",
//                             colocateWith = first_var)
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
//                         tf.cast(beta1_power, _v.dataType.base_dtype),
//                         tf.cast(beta2_power, _v.dataType.base_dtype),
//                         tf.cast(lr_t, _v.dataType.base_dtype),
//                         tf.cast(beta1_t, _v.dataType.base_dtype),
//                         tf.cast(beta2_t, _v.dataType.base_dtype),
//                         tf.cast(epsilon_t, _v.dataType.base_dtype),
//                         grad, use_locking = use_locking).op!!
//  }
//
//  override fun finish(update_ops: MutableList<Op>, name: String): Op {
//    tf.controlDependencies(update_ops) {
//      val (beta1_power, beta2_power) = get_beta_accumulators()
//      tf.colocateWith(beta1_power) {
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
//      t2(get_non_slot_variable("beta1_power"),
//             get_non_slot_variable("beta2_power"))
//
//}