package wumo.sim.algorithm.tensorflow.training

import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.gradients
import wumo.sim.algorithm.tensorflow.ops.group
import wumo.sim.algorithm.tensorflow.ops.variable
import wumo.sim.algorithm.tensorflow.tf

/**
 * This class defines the API to add Ops to train a model.  You never use this
 * class directly, but instead instantiate one of its subclasses such as
 * **GradientDescentOptimizer**, **AdagradOptimizer**, or **MomentumOptimizer**.
 */
abstract class Optimizer(val use_locking: Boolean, val name: String) {
  val slots = mutableMapOf<String, MutableMap<Variable, Variable>>()
  val non_slot_dict = mutableMapOf<String, Variable>()
  
  fun minimize(loss: Tensor, var_list: List<Variable>? = null, name: String = ""): Operation {
    val grads_and_vars = compute_gradients(loss, var_list)
    val vars_with_grad = grads_and_vars.map { (g, v) -> v }
    return apply_gradients(grads_and_vars, name = name)
  }
  
  fun compute_gradients(loss: Tensor, var_list: List<Variable>?): List<Pair<Tensor, Variable>> {
    val var_list = var_list ?: tf.trainables
    val grads = tf.gradients(loss, var_list)
    return grads.zip(var_list)
  }
  
  fun apply_gradients(grads_and_vars: List<Pair<Tensor, Variable>>, name: String): Operation {
    val name = if (name.isEmpty()) this.name else name
    val var_list = grads_and_vars.map { (g, v) -> v }
    create_slots(var_list)
    val update_ops = mutableListOf<Operation>()
    with(tf) {
      subscope(name) {
        prepare()
        for ((grad, v) in grads_and_vars)
          ctx.colocate_with(v) {
            update_ops += apply_dense(grad, v)
          }
        val apply_updates = finish(update_ops, borrowParentName())
        tf.train_ops += apply_updates
        return apply_updates
      }
    }
  }
  
  open fun create_slots(var_list: List<Variable>) {}
  
  abstract fun prepare()
  
  //TODO sparse IndexedSlices
  abstract fun apply_dense(grad: Tensor, v: Variable): Operation
  
  open fun finish(update_ops: MutableList<Operation>, name: String) =
      tf.group(update_ops, name)
  
  open fun slot_dict(slot_name: String): MutableMap<Variable, Variable> {
    return slots.compute(slot_name) { _, named_slots ->
      named_slots ?: mutableMapOf()
    }!!
  }
  
  fun zero_slot(v: Variable, slot_name: String, op_name: String): Variable {
    val named_slots = slot_dict(slot_name)
    return named_slots.compute(v) { _, slot_variable ->
      slot_variable ?: create_zeros_slot(v, op_name)
      
    }!!
  }
  
  /**Add an extra variable, not associated with a slot.*/
  open fun create_non_slot_variable(initial_value: Any, name: String, colocate_with: Variable) =
      non_slot_dict.compute(name) { _, v ->
        v ?: tf.ctx.colocate_with(colocate_with) {
          tf.variable(initial_value, name = name, trainable = false)
        }
      }
  
  protected fun get_non_slot_variable(name: String): Variable {
    val non_slot = non_slot_dict[name]
    return non_slot!!
  }
}