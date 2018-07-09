package wumo.sim.algorithm.tensorflow.training

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.gradients
import wumo.sim.algorithm.tensorflow.ops.group
import wumo.sim.algorithm.tensorflow.tf

/**
 * This class defines the API to add Ops to train a model.  You never use this
 * class directly, but instead instantiate one of its subclasses such as
 * **GradientDescentOptimizer**, **AdagradOptimizer**, or **MomentumOptimizer**.
 */
abstract class Optimizer(val use_locking: Boolean, val name: String) {
  val slots = mutableMapOf<String, MutableMap<Tensor, Tensor>>()
  
  fun minimize(loss: Tensor, var_list: List<Tensor>? = null, name: String = ""): Operation {
    val grads_and_vars = compute_gradients(loss, var_list)
    val vars_with_grad = grads_and_vars.map { (g, v) -> v }
    return apply_gradients(grads_and_vars, name = name)
  }
  
  fun compute_gradients(loss: Tensor, var_list: List<Tensor>?): List<Pair<Tensor, Tensor>> {
    val var_list = if (var_list == null) tf.trainables else var_list
    val grads = tf.gradients(loss, var_list)
    return grads.zip(var_list)
  }
  
  fun apply_gradients(grads_and_vars: List<Pair<Tensor, Tensor>>, name: String): Operation {
    val name = if (name.isEmpty()) this.name else name
    val var_list = grads_and_vars.map { (g, v) -> v }
    create_slots(var_list)
    val update_ops = mutableListOf<Operation>()
    with(tf) {
      subscope(name) {
        prepare()
        for ((grad, v) in grads_and_vars)
          update_ops += apply_dense(grad, v)
        val apply_updates = finish(update_ops, borrowParentName())
        tf.train_ops += apply_updates
        return apply_updates
      }
    }
  }
  
  open fun create_slots(var_list: List<Tensor>) {}
  
  abstract fun prepare()
  
  abstract fun apply_dense(grad: Tensor, v: Tensor): Operation
  
  open fun finish(update_ops: MutableList<Operation>, name: String) =
      tf.group(update_ops, name)
  
  open fun slot_dict(slot_name: String): MutableMap<Tensor, Tensor> {
    return slots.compute(slot_name) { _, named_slots ->
      named_slots ?: mutableMapOf()
    }!!
  }
  
  fun zero_slot(v: Tensor, slot_name: String, op_name: String): Tensor {
    val named_slots = slot_dict(slot_name)
    return named_slots.compute(v) { _, slot_variable ->
    
    }!!
  }
  
  open fun create_non_slot_variable(initial_value: Any, name: String, colocate_with: Tensor) {
  
  }
}