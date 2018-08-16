package wumo.sim.tensorflow.training
//
//import wumo.sim.tensorflow.ops.Op
//import wumo.sim.tensorflow.ops.Output
//import wumo.sim.tensorflow.ops.ops
//import wumo.sim.tensorflow.ops.variables.Variable
//import wumo.sim.tensorflow.tf
//import wumo.sim.util.tuple2
//
///**
// * This class defines the API to add Ops to train a model.  You never use this
// * class directly, but instead instantiate one of its subclasses such as
// * **GradientDescentOptimizer**, **AdagradOptimizer**, or **MomentumOptimizer**.
// */
//abstract class Optimizer(val use_locking: Boolean, val name: String) {
//
//  val slots = mutableMapOf<String, MutableMap<Variable, Variable>>()
//  val non_slot_dict = mutableMapOf<String, Variable>()
//
//  fun minimize(loss: Output, var_list: Collection<Variable>? = null, name: String = ""): Op {
//    val grads_and_vars = compute_gradients(loss, var_list)
//    val vars_with_grad = grads_and_vars.map { (g, v) -> v }
//    return apply_gradients(grads_and_vars, name = name)
//  }
//
//  fun compute_gradients(loss: Output, var_list: Collection<Variable>?): List<tuple2<Output, Variable>> {
//    val var_list = var_list ?: tf.trainables
//    val grads = tf.gradients(loss, var_list)
//    return grads.zip(var_list)
//  }
//
//  fun apply_gradients(grads_and_vars: List<tuple2<Output, Variable>>, name: String = ""): Op {
//    val name = if (name.isEmpty()) this.name else name
//    val var_list = grads_and_vars.map { (g, v) -> v }
//    create_slots(var_list)
//    val update_ops = mutableListOf<Op>()
//    ops.nameScope(name) {
//      prepare()
//      for ((grad, v) in grads_and_vars)
//        ops.nameScope("update_$name") {
//          ops.colocateWith(v.op!!) {
//            update_ops += apply_dense(grad, v)
//          }
//        }
//      val apply_updates = finish(update_ops, name)
//      tf.train_ops += apply_updates
//      return apply_updates
//    }
//  }
//
//  open fun create_slots(var_list: List<Variable>) {}
//
//  abstract fun prepare()
//
//  //TODO sparse IndexedSlices
//  abstract fun apply_dense(grad: Output, v: Variable): Op
//
//  open fun finish(update_ops: MutableList<Op>, name: String) =
//      tf.group(update_ops, name)
//
//  open fun slot_dict(slot_name: String): MutableMap<Variable, Variable> {
//    return slots.compute(slot_name) { _, named_slots ->
//      named_slots ?: mutableMapOf()
//    }!!
//  }
//
//  fun zero_slot(v: Variable, slot_name: String, op_name: String): Variable {
//    val named_slots = slot_dict(slot_name)
//    return named_slots.compute(v) { _, slot_variable ->
//      slot_variable ?: create_zeros_slot(v, op_name)
//
//    }!!
//  }
//
//  /**
//   * Return a slot named `name` created for `var` by the Optimizer.
//
//  Some `Optimizer` subclasses use additional variables.  For example
//  `Momentum` and `Adagrad` use variables to accumulate updates.  This method
//  gives access to these `Variable` objects if for some reason you need them.
//
//  Use `get_slot_names()` to get the list of slot names created by the
//  `Optimizer`.
//
//   * @param v: A variable passed to `minimize()` or `apply_gradients()`.
//   * @param name: A string.
//
//   * @return: The `Variable` for the slot if it was created, `None` otherwise.
//   */
//  open fun get_slot(v: Variable, name: String): Variable {
//    val named_slots = slots[name]
//    return named_slots!![v]!!
//  }
//
//  /**Add an extra variable, not associated with a slot.*/
//  open fun create_non_slot_variable(initial_value: Any, name: String, colocateWith: Variable) =
//      non_slot_dict.compute(name) { _, v ->
//        v ?: tf.colocateWith(colocateWith) {
//          tf.variable(initial_value, name = name, trainable = false)
//        }
//      }
//
//  protected fun get_non_slot_variable(name: String): Variable {
//    val non_slot = non_slot_dict[name]
//    return non_slot!!
//  }
//}