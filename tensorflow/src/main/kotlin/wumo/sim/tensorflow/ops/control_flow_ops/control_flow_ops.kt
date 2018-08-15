package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.util.a
import kotlin.collections.component1
import kotlin.collections.component2

object control_flow_ops {
  /** Returns `true` if the provided op is within a cond statement. */
  val Op.isInCond: Boolean
    get() = controlFlowContext?.condContext() != null
  
  /** Returns `true` if the provided op is within a while loop statement. */
  val Op.isInWhileLoop: Boolean
    get() = controlFlowContext?.whileContext() != null
  
  /** Returns `true` if the provided op is within an XLA control flow context. */
  val Op.isInXLAContext: Boolean
    get() = run {
      val xlaCompile =
          try {
            attrBool("_XlaCompile")
          } catch (_: IllegalArgumentException) {
            false
          }
      xlaCompile || controlFlowContext?.xlaContext() != null
    }
  
  /** Returns `true` if and only if the provided op is a switch op. */
  internal fun isSwitch(op: Op): Boolean = op.opType == "Switch" || op.opType == "RefSwitch"
  
  /** Returns `true` if and only if the provided op is a merge op. */
  internal fun isMerge(op: Op): Boolean = op.opType == "Merge" || op.opType == "RefMerge"
  
  /** Returns `true` if and only if the provided op is a switch op for a conditional. */
  internal fun isCondSwitch(op: Op): Boolean =
      if (!isSwitch(op) || op.outputs.isEmpty())
        false
      else
      // Switch nodes are not part of the "cond" control flow context that they represent, and so we consider the
      // consumers of its outputs to determine if it is a "cond" switch or not. A switch is a "cond" switch if and only
      // if all its consumers are in "cond" contexts.
        op.outputs.all {
          it.consumers.all { c ->
            var context = c.controlFlowContext
            if (isLoopEnter(c))
              context = context?.outerContext
            context is CondContext
          }
        }
  
  /** Returns `true` if and only if the provided op is a merge op for a conditional. */
  internal fun isCondMerge(op: Op): Boolean =
      if (!isMerge(op) || op.inputs.isEmpty())
        false
      else
      // Merge nodes are not part of the "cond" control flow context that they represent, and so we consider their
      // inputs to determine if they are "cond" merges or not. A merge is a "cond" merge if and only if all its inputs
      // are in "cond" contexts.
        op.inputs.all { i ->
          val context = getOutputContext(i.op!!)
          context is CondContext
        }
  
  /** Returns `true` if and only if the provided op is a loop invariant. */
  internal fun isLoopEnter(op: Op): Boolean = op.opType == "Enter" || op.opType == "RefEnter"
  
  /** Returns `true` if and only if the provided op is a loop exit op. */
  internal fun isLoopExit(op: Op): Boolean = op.opType == "Exit" || op.opType == "RefExit"
  
  /** Returns `true` if and only if the provided op is a switch op for a while loop. */
  internal fun isLoopSwitch(op: Op): Boolean =
      isSwitch(op) &&
          op.controlFlowContext is WhileContext &&
          !isCondSwitch(op)
  
  /** Returns `true` if and only if the provided op is a merge op for a while loop. */
  internal fun isLoopMerge(op: Op): Boolean =
      isMerge(op) &&
          op.controlFlowContext is WhileContext &&
          !isCondMerge(op)
  
  /** Returns `true` if and only if the provided op is a constant loop invariant. */
  internal fun isLoopConstantEnter(op: Op): Boolean = run {
    isLoopEnter(op) && op.attrBool("is_constant")
  }
  
  private val identityOpTypes = setOf("Identity", "RefIdentity", "Switch", "RefSwitch")
  /** Returns the enter op if we can infer `value` to be a loop invariant. Otherwise, returns [[None]]. */
  internal fun getLoopConstantEnter(value: Output): Op? {
    var op = value.op!!
    while (op.opType in identityOpTypes)
      op = op.inputs[0].op!!
    return op.takeIf { isLoopConstantEnter(it) }
  }
  
  /** Returns the control flow context for the outputs of an op. */
  internal fun getOutputContext(op: Op): ControlFlowContext? = run {
    val context = op.controlFlowContext
    if (isLoopExit(op))
      context?.outerContext
    else
      context
  }
  
  /**
   * @see[ControlFlowContext.xlaContext]
   */
  internal fun getContainingXLAContext(context: ControlFlowContext?) =
      context?.xlaContext()
  
  /**
   * @see[ControlFlowContext.whileContext]
   */
  internal fun getContainingWhileContext(context: ControlFlowContext?,
                                         stopContext: ControlFlowContext? = null) =
      context?.whileContext(stopContext)
  
  /**
   * @see[ControlFlowContext.condContext]
   */
  internal fun getContainingCondContext(context: ControlFlowContext?) =
      context?.condContext()
  
  /** Returns `true` if `maybeContainingContext` is or contains `context`. */
  internal fun isContainingContext(context: ControlFlowContext?,
                                   maybeContainingContext: ControlFlowContext?): Boolean {
    if (context == null && maybeContainingContext == null) return false
    var currentContext = context
    while (currentContext !== maybeContainingContext)
      currentContext = currentContext?.outerContext ?: return false
    return true
  }
  
  /**
   * Returns whether [inputOp] can be used from [op]s context.
   
   * Conceptually, only inputs from op's while context or any ancestor while
   * context (including outside of any context) are valid. In practice, there are
   * many other edge cases as well.
   */
  fun checkInputFromValidContext(op: Op, inputOp: Op) {
    val opContext = op.controlFlowContext
    val inputContext = getOutputContext(inputOp)
    
    val errorMsg = when (inputContext) {
      null -> null  //input_op isn't in a control flow context.
      opContext -> null //input_op is in the same context as op.
      else -> {
        val whileContext = opContext?.whileContext()
        val inputWhileContext = inputContext.whileContext()
        when {
          whileContext == null -> {
            when {
              inputWhileContext == null -> null
              // Neither `op` nor `inputOp` is in a while loop, but one or both are in conditionals. We allow this,
              // although execution will fail if the branch corresponding to the `inputOp`'s conditional context is not
              // taken.
              isLoopEnter(op) || isSwitch(op) -> null
              // The while loop building code clears the context for enter nodes, and the conditional context add value
              // code clears the context for switch nodes.
              else -> "Cannot use '${inputOp.name}' as input to '${op.name}' because '${inputOp.name}' is in a while loop."
            }
          }
          
          isContainingContext(whileContext, inputWhileContext) -> null
          // `inputOp` is in a while loop which contains `op`'s while loop (or not in a while loop at all).
          whileContext.gradState != null &&
              isContainingContext(whileContext.gradState.forwardContext,
                                  inputWhileContext) -> null
          // `op` is in a gradient context and `inputOp` is in the associated forward pass context or an ancestor
          // thereof. This case is needed to build while loop gradients. Note that we theoretically also need this
          // case for custom gradient functions that close over tensors from ancestor contexts, but this has not been
          // verified yet.
          whileContext.gradState != null &&
              whileContext.gradState.forwardContext === inputWhileContext?.outerContext -> null
          // `op` is in a gradient context and `inputOp` is in a child of the associated forward pass context. This
          // case is needed for the gradients of while loops with conditionals.
          inputWhileContext?.gradState != null &&
              inputWhileContext.gradState.forwardContext === whileContext -> null
          // `inputOp` is in the gradient context of `op`'s context. This case is needed when the gradient of a while
          // loop gradient is requested (this will eventually fail unless there is a `stopGradient` op or similar).
          inputWhileContext?.gradState != null &&
              inputContext.gradState?.forwardContext?.gradState != null &&
              inputContext.gradState!!.forwardContext.gradState!!.forwardContext === whileContext -> null
          // `inputOp` is in the gradient gradient context of `op`'s context. This case is needed when the gradient of
          // a while loop gradient is requested (this will eventually fail unless there is a `stopGradient` op or
          // similar).
          else -> "Cannot use '${inputOp.name}' as input to '${op.name}' because they are in different while loops."
        }
      }
    }
    if (errorMsg != null) throw InvalidArgumentException(errorMsg)
  }
  
  private fun groupControlDeps(deps: List<Op>, name: String = "noOp") = run {
    tf.control_dependencies(deps) {
      tf._noOp(name)
    }
  }
  
  /**
   * @see [_merge]
   */
  private fun ref_merge(inputs: Array<Output>, name: String): Array<Output> =
      tf.colocate_with_tensors(inputs) {
        tf._refMerge(inputs, name)
      }
  
  interface API {
    /**
     * @see [switch]
     */
    fun _switchRefOrTensor(data: Output,
                           pred: Output,
                           name: String = "Switch"): Array<Output> =
        tf.colocate_with(data, ignore_existing = true) {
          if (data.dtype.isRefType)
            tf._refSwitch(data, pred, name)
          else
            tf._switch(data, pred, name)
        }
    
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
     * @param output_tensor: A `Output` or `IndexedSlices` that will be returned.
     * @param name: (Optional) A name for this operation.
    
    Returns:
    Same as `output_tensor`.
     */
    fun with_dependencies(vararg dependencies: Op,
                          output_tensor: Output,
                          name: String = "control_dependency"): Output =
        tf.name_scope(name) {
          tf.colocate_with(output_tensor) {
            tf.control_dependencies(*dependencies) {
              tf._identity(output_tensor, name = tf.currentNameScope)
              //TODO indexedSlices
            }
          }
        }
    
    fun identity(data: Output, name: String): Output {
      return if (data.dtype.isRefType)
        tf._refIdentity(data, name)
      else
        tf._identity(data, name)
      //TODO indexedSlice
    }
    
    fun group(inputs: List<Any>, name: String = "group_deps"): Op {
      val ops_on_device = mutableMapOf<String, MutableList<Op>>()
      for (input in inputs) {
        val op = when (input) {
          is Op -> input
          is Variable -> input.initializer
          is Output -> input.op
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
        return tf.on_device(dev) {
          groupControlDeps(deps, name)
        }
      }
      val all_deps = mutableListOf<Op>()
      return tf.name_scope(name) {
        for ((dev, deps) in ops_on_device) {
          tf.on_device(dev) {
            all_deps += groupControlDeps(deps)
          }
        }
        groupControlDeps(all_deps, tf.currentNameScope)
      }
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
    fun cond(pred: Output,
             true_fn: () -> Any,
             false_fn: () -> Any,
             name: String = "cond"): Output =
        tf.name_scope(name) {
          val (p_2, p_1) = tf._switch(pred, pred)
          val pivot_1 = tf._identity(p_1, name = "switch_t")
          val pivot_2 = tf._identity(p_2, name = "switch_f")
          val pred = tf._identity(pred, name = "pred_id")
          //Disable the fetching of tensors that are only on one branch of cond.
          for (tensor in a(p_1, p_2, pivot_1, pivot_2, pred))
            tf.currentGraph.prevent_fetching(tensor.op!!)
          //Build the graph for the true branch in a new context.
          val res_t = tf.condContext(pred, pivot_1, branch = 1) {
            it.buildCondBranch(true_fn)
          }
          val res_f = tf.condContext(pred, pivot_2, branch = 0) {
            it.buildCondBranch(false_fn)
          }
//    val res_t = buildCondBranch(pred, pivot_1, 1, true_fn)
//     = buildCondBranch(pred, pivot_2, 0, false_fn)
          merge(res_t, res_f)[0]
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
     * This op handles both `Output`s and `IndexedSlices`. If inputs has a mix of
     * `Output`s and `IndexedSlices`, all inputs are converted to IndexedSlices
     * before merging.
     * @param inputs The input tensors, at most one of which is available.
     * @param name A name for this operation (optional).
     * @return A tuple containing the chosen input tensor and its index in `inputs`.
     */
    fun merge(vararg inputs: Output, name: String = "Merge"): Array<Output> {
      return if (inputs.all { it.dtype.isRefType })
        ref_merge(inputs as Array<Output>, name)
      else
        tf._merge(inputs as Array<Output>, name)
      //TODO handle sparseTensor indexedSlices
    }
  }
}