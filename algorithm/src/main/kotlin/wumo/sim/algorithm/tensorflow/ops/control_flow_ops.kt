package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.util.a

class CondContext(val pred: Tensor,
                  val pivot: Tensor,
                  val branch: Int) : ControlFlowContext {
  override fun addOp(op: Operation) {
    if (op.inputs.isEmpty()) {
      //TODO Remove any external control dependency on this op
      TODO()//add control input
    } else {
      for (i in 0 until op.inputs.size) {
        val x = op.inputs[i]
        val real_x = addValue(x)
        if (real_x != x)
          op.update_input(i, real_x)
      }
    }
    TODO("not implemented")
  }
  
  /**Add `val` to the current context and its outer context recursively.*/
  private fun addValue(x: Tensor): Tensor {
    TODO("not implemented")
  }
}

fun TF.group(inputs: List<Any>, name: String = "group_deps"): Operation {
  val ops_on_device = mutableMapOf<String, MutableList<Operation>>()
  for (input in inputs) {
    val op = when (input) {
      is Operation -> input
      is Variable -> input.initializer_op.op
      else -> throw IllegalArgumentException("unsupported ${input::class.java}")
    }
    val dev = op.device
    ops_on_device.compute(dev) { _, list ->
      val list = list ?: mutableListOf()
      list += op
      list
    }
  }
  if (ops_on_device.size == 1) {
    val (dev, deps) = ops_on_device.entries.first()
    ctx.with_device(dev) {
      return noOpDep(deps, name)
    }
  }
  val all_deps = mutableListOf<Operation>()
  subscope(name) {
    for ((dev, deps) in ops_on_device) {
      ctx.with_device(dev) {
        all_deps += noOpDep(deps)
      }
    }
  }
  return noOpDep(all_deps, name)
}

/**
 * Return `true_fn()` if the predicate `pred` is true else `false_fn()`.
 *
 * `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
 * `false_fn` must have the same non-zero number and type of outputs.
 
 * Note that the conditional execution applies only to the operations defined in
 * `true_fn` and `false_fn`. Consider the following simple program:
 * @param pred  A scalar determining whether to return the result of `true_fn` or `false_fn`.
 * @param true_fn The callable to be performed if pred is true.
 * @param false_fn he callable to be performed if pred is false.
 * @return  Tensors returned by the call to either `true_fn` or `false_fn`. If the
 * callables return a singleton list, the element is extracted from the list.
 */
fun TF.cond(pred: Tensor,
            true_fn: () -> Tensor,
            false_fn: () -> Tensor,
            name: String = "cond"): Tensor {
  val (p_2, p_1) = switch(pred, pred)
  val pivot_1 = identity(p_1, name = "switch_t")
  val pivot_2 = identity(p_2, name = "switch_f")
  val pred = identity(pred, name = "pred_id")
  //Disable the fetching of tensors that are only on one branch of cond.
  for (tensor in a(p_1, p_2, pivot_1, pivot_2, pred))
    g.prevent_fetching(tensor.op)
  
  //Build the graph for the true branch in a new context.
  val res_t = condCtx(pred, pivot_1, branch = 1) {
    buildCondBranch(true_fn)
  }
  val res_f = condCtx(pred, pivot_2, branch = 0) {
    buildCondBranch(false_fn)
  }
  return merge(res_t, res_f)[0]
}

fun TF.buildCondBranch(fn: () -> Tensor): Tensor {
  val t = fn()
  val cf: CondContext = ctx.control_flow_ctx as CondContext
  return switchRefOrTensor(t, cf.pred)[cf.branch]
}

/**
 * Returns the value of an available element of `inputs`.
 *
 * This op tests each of the tensors in `inputs` in turn to determine if any of
 * them is available. If it finds an available tensor, it returns it and its
 * index in `inputs`.
 
 * It is an error if more than one tensor in `inputs` is available. If no tensor
 * in `inputs` is available, the returned tensor and index are not set.
 *
 * This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
 * `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
 * before merging.
 * @param inputs The input tensors, at most one of which is available.
 * @param name A name for this operation (optional).
 * @return A tuple containing the chosen input tensor and its index in `inputs`.
 */
fun TF.merge(vararg inputs: Tensor, name: String = "Merge"): Array<Tensor> {
  return if (inputs.all { it.dtype.is_ref_dytpe })
    ref_merge(inputs as Array<Tensor>, name)
  else
    _merge(inputs as Array<Tensor>, name)
  //TODO handle sparseTensor indexedSlices
}

/**
 * Forwards the value of an available tensor from `inputs` to `output`.
 *
 * `Merge` waits for at least one of the tensors in `inputs` to become available.
 *
 * It is usually combined with `Switch` to implement branching.
 *
 * `Merge` forwards the first tensor to become available to `output`, and sets
 * `value_index` to its index in `inputs`.
 *
 * @param inputs A list of at least 1 `Tensor` objects with the same type.
 * The input tensors, exactly one of which will become available.
 * @param name A name for the operation (optional).
 * @return A tuple of `Tensor` objects (output, value_index).
 */
private fun TF._merge(inputs: Array<Tensor>, name: String = "Merge"): Array<Tensor> {
  val op = g.nodeBuilder("Merge", ctx.getUniqueFullName(name))
      .addInputList(inputs)
      .build()
  val output = Tensor(op, 0)
  val value_index = Tensor(op, 1)
  return a(output, value_index)
}

/**
 * @see [_merge]
 */
private fun TF.ref_merge(inputs: Array<Tensor>, name: String): Array<Tensor> {
  val op = g.nodeBuilder("RefMerge", ctx.getUniqueFullName(name))
      .addInputList(inputs)
      .build()
  val output = Tensor(op, 0)
  val value_index = Tensor(op, 1)
  return a(output, value_index)
}

/**
 * @see [switch]
 */
private fun TF.switchRefOrTensor(data: Tensor,
                                 pred: Tensor,
                                 name: String = "Switch"): Array<Tensor> {
  tf.colocate_with(data) {
    if (data.dtype.is_ref_dytpe)
      return ref_switch(data, pred, name)
    return switch(data, pred, name)
  }
}

/**
 * Forwards [data] to an output determined by [pred].
 *
 * If `pred` is false, the `data` input is forwarded to the first output.
 * Otherwise, the data goes to the second output.
 
 * This op handles `Tensor`s and `IndexedSlices`.
 * @param data The tensor to be forwarded to the appropriate output.
 * @param pred A scalar that specifies which output port will receive data.
 * @param dtype Optional element type for the returned tensor. If missing,
 * the type is inferred from the type of `value`.
 * @param name A name for this operation (optional).
 * @return `(output_false, output_true)`: If `pred` is true, data will be forwarded
 * to `output_true`, otherwise it goes to `output_false`.
 */
fun TF.switch(data: Tensor, pred: Tensor, name: String = "Switch"): Array<Tensor> {
  val op = g.nodeBuilder("Switch", ctx.getUniqueFullName(name))
      .addInput(data)
      .addInput(pred)
      .build()
  //TODO handle IndexedSlices and SparseTensor
  val output_false = Tensor(op, 0)
  val output_true = Tensor(op, 1)
  return a(output_false, output_true)
}

fun TF.ref_switch(data: Tensor, pred: Tensor, name: String): Array<Tensor> {
  val op = g.nodeBuilder("RefSwitch", ctx.getUniqueFullName(name))
      .addInput(data)
      .addInput(pred)
      .build()
  //TODO handle IndexedSlices and SparseTensor
  val output_false = Tensor(op, 0)
  val output_true = Tensor(op, 1)
  return a(output_false, output_true)
}
