package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.*
import wumo.sim.util.a

abstract class ControlFlowContext {
  var outer_context = tf.control_flow_context
  val values = hashSetOf<String>()
  val external_values = hashMapOf<String, Tensor>()
  abstract fun addOp(op: Operation)
}

class CondContext(val pred: Tensor,
                  val pivot: Tensor,
                  val branch: Int) : ControlFlowContext() {
  
  init {
    //Values considered to have been already seen in this context. pred is not
    //included in this context.
    values += pred.name
    external_values[pred.name] = pred
    values += pivot.name
  }
  
  override fun addOp(op: Operation) {
    if (op.inputs.isEmpty()) {
      _removeExternalControlEdges(op)
      op.addControlInput(pivot.op!!)
    } else {
      for (i in 0 until op.inputs.size) {
        val x = op.inputs[i]
        val real_x = addValue(x)
        if (real_x != x)
          op.update_input(i, real_x)
        _removeExternalControlEdges(op)
        if (op.graph.is_function(op.opType) || op.opType == "SymbolicGradient")
          op.addControlInput(pivot.op!!)
      }
    }
    //Mark op's outputs as seen by this context and any outer contexts.
    val output_names = op.outputs.map { it.name }
    var ctxt: ControlFlowContext? = this
    while (ctxt != null) {
      ctxt.values.addAll(output_names)
      ctxt = ctxt.outer_context
    }
    if (outer_context != null || !isLoopExit(op))
      op.graph.prevent_fetching(op)
  }
  
  private fun _removeExternalControlEdges(op: Operation) {
    //TODO Remove any external control dependency on this op
  }
  
  /**Add `val` to the current context and its outer context recursively.*/
  private fun addValue(v: Tensor): Tensor {
    return if (v.name in values) {
      //Use the real value if it comes from outer context. This is needed in
      //particular for nested conds.
      external_values[v.name] ?: v
    } else {
      var result = v
      values += v.name
      if (outer_context != null) {
        result = (outer_context as CondContext).addValue(v)
        values += result.name
        external_values[result.name] = result
      }
      tf.control_dependencies {
        result = tf.switchRefOrTensor(result, pred)[branch]
      }
      result.op!!.graph.prevent_fetching(result.op!!)
      
      values += result.name
      external_values[v.name] = result
      result
    }
  }
  
  fun buildCondTensor(v: Any): Tensor {
    return when (v) {
      is Operation -> {//Use pivot as the proxy for this op.
        with_dependencies(v, output_tensor = pivot)
      }
      is IndexedSlices, is SparseTensor -> {
        TODO()
      }
      else -> processOutputTensor(v as Tensor)
    }
  }
  
  /**Add the subgraph defined by fn() to the graph.*/
  fun buildCondBranch(fn: () -> Tensor): Tensor {
    val original_result = fn()
    val result = buildCondTensor(original_result)
    return result
  }
  
  private fun processOutputTensor(v: Tensor): Tensor {
    var real_v = v
    if (v.name !in values) {
      values += v.name
      real_v = tf.switchRefOrTensor(v, pred)[branch]
      external_values[v.name] = real_v
    } else {
      val external_v = external_values[v.name]
      if (external_v != null)
        real_v = external_v
    }
    return real_v
  }
}

/**Return true if `op` is an Exit.*/
fun isLoopExit(op: Operation) = op.opType == "Exit" || op.opType == "RefExit"

/**
Produces the content of `output_tensor` only after `dependencies`.

In some cases, a user may want the output of an operation to be
consumed externally only after some other dependencies have run
first. This function ensures returns `output_tensor`, but only after all
operations in `dependencies` have run. Note that this means that there is
no guarantee that `output_tensor` will be evaluated after any `dependencies`
have run.

See also @{tf.tuple$tuple} and @{tf.group$group}.

Args:
 * @param dependencies: Iterable of operations to run before this op finishes.
 * @param output_tensor: A `Tensor` or `IndexedSlices` that will be returned.
 * @param name: (Optional) A name for this operation.

Returns:
Same as `output_tensor`.
 */
fun with_dependencies(vararg dependencies: Operation,
                      output_tensor: Tensor,
                      name: String = "control_dependency"): Tensor {
  with(tf) {
    name_scope(name) {
      colocate_with(output_tensor) {
        control_dependencies(*dependencies) {
          return _identity(output_tensor, name = name)
          //TODO indexedSlices
        }
      }
    }
  }
}

fun TF._identity(data: Tensor, name: String): Tensor {
  return if (data.dtype.is_ref_dytpe)
    refIdentity(data, name)
  else
    identity(data, name)
  //TODO indexedSlice
}

fun TF.group(inputs: List<Any>, name: String = "group_deps"): Operation {
  val ops_on_device = mutableMapOf<String, MutableList<Operation>>()
  for (input in inputs) {
    val op = when (input) {
      is Operation -> input
      is Variable -> input.initializer_op.op
      is Tensor -> input.op
      else -> throw IllegalArgumentException("unsupported ${input::class.java}")
    }
    val dev = op!!.device
    ops_on_device.compute(dev) { _, list ->
      val list = list ?: mutableListOf()
      list += op
      list
    }
  }
  if (ops_on_device.size == 1) {
    val (dev, deps) = ops_on_device.entries.first()
    ctxNs.with_device(dev) {
      return noOpDep(deps, name)
    }
  }
  val all_deps = mutableListOf<Operation>()
  name_scope(name) {
    for ((dev, deps) in ops_on_device) {
      ctxNs.with_device(dev) {
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
  name_scope(name) {
    val (p_2, p_1) = switch(pred, pred)
    val pivot_1 = identity(p_1, name = "switch_t")
    val pivot_2 = identity(p_2, name = "switch_f")
    val pred = identity(pred, name = "pred_id")
    //Disable the fetching of tensors that are only on one branch of cond.
    for (tensor in a(p_1, p_2, pivot_1, pivot_2, pred))
      g.prevent_fetching(tensor.op!!)
    
    //Build the graph for the true branch in a new context.
    val res_t = condContext(pred, pivot_1, branch = 1) {
      it.buildCondBranch(true_fn)
    }
    val res_f = condContext(pred, pivot_2, branch = 0) {
      it.buildCondBranch(false_fn)
    }
//    val res_t = buildCondBranch(pred, pivot_1, 1, true_fn)
//     = buildCondBranch(pred, pivot_2, 0, false_fn)
    return merge(res_t, res_f)[0]
  }
}

fun TF.buildCondBranch(pred: Tensor, pivot: Tensor, branch: Int, fn: () -> Tensor): Tensor {//TODO control deped on pivot Tensor
  condContext(pred, pivot, branch) {
    val t = fn()
    return switchRefOrTensor(t, pred)[branch]
  }
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
  val op = g.nodeBuilder("Merge", ctxNs.getUniqueFullName(name))
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
  val op = g.nodeBuilder("RefMerge", ctxNs.getUniqueFullName(name))
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
  val op = g.nodeBuilder("Switch", ctxNs.getUniqueFullName(name))
      .addInput(data)
      .addInput(pred)
      .build()
  //TODO handle IndexedSlices and SparseTensor
  val output_false = Tensor(op, 0)
  val output_true = Tensor(op, 1)
  return a(output_false, output_true)
}

fun TF.ref_switch(data: Tensor, pred: Tensor, name: String): Array<Tensor> {
  val op = g.nodeBuilder("RefSwitch", ctxNs.getUniqueFullName(name))
      .addInput(data)
      .addInput(pred)
      .build()
  //TODO handle IndexedSlices and SparseTensor
  val output_false = Tensor(op, 0)
  val output_true = Tensor(op, 1)
  return a(output_false, output_true)
}
