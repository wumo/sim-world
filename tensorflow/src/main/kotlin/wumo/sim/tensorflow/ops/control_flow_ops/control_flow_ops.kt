package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.util.a
import kotlin.collections.component1
import kotlin.collections.component2

object control_flow_ops {
  /** Returns `true` if the provided op is within a cond statement. */
  val Op.isInCond: Boolean
    get() = controlFlowContext?.condContext != null
  
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
      context?.condContext
  
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
              isContainingContext(whileContext.gradState?.forwardContext,
                                  inputWhileContext) -> null
          // `op` is in a gradient context and `inputOp` is in the associated forward pass context or an ancestor
          // thereof. This case is needed to build while loop gradients. Note that we theoretically also need this
          // case for custom gradient functions that close over tensors from ancestor contexts, but this has not been
          // verified yet.
          whileContext.gradState != null &&
              whileContext.gradState?.forwardContext === inputWhileContext?.outerContext -> null
          // `op` is in a gradient context and `inputOp` is in a child of the associated forward pass context. This
          // case is needed for the gradients of while loops with conditionals.
          inputWhileContext?.gradState != null &&
              inputWhileContext.gradState?.forwardContext === whileContext -> null
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
    tf.controlDependencies(deps) {
      tf._noOp(name)
    }
  }
  
  /**
   * @see [_merge]
   */
  private fun ref_merge(inputs: List<Output>, name: String): Array<Output> =
      tf.colocateWith(inputs) {
        tf._refMerge(inputs, name)
      }
  
  /** Creates an op that forwards [input] to the output port determined by [predicate], while making sure the new op is
   * colocated with [input].
   *
   * If [predicate] is `true`, then [input] is forwarded to `outputTrue`. Otherwise, it goes to `outputFalse`.
   *
   * @param  input     Tensor to be forwarded to the appropriate output.
   * @param  predicate Scalar boolean tensor that specifies which output port will receive [input].
   * @param  name      Name for the created op.
   * @return Tuple containing `outputFalse` and `outputTrue`, in that order.
   */
  fun _switchRefOrTensor(input: Output,
                         predicate: Output,
                         name: String = "Switch"): Array<Output> {
    // The device colocation below addresses the following scenario:
    //
    // Assume you execute Optimizer.applyGradients() in a branch of a cond() and:
    //   1. The update op is created inside a `Op.colocateWith(Set(var.op)) { }` block.
    //   2. Some tensor `data` is captured and a switch is created in a `Op.colocateWith(Set(data.op)) { }` block.
    //
    // Op.colocateWith(Set(var.op)) {
    //   Op.colocateWith(Set(data.op)) {
    //     op = ...
    //   }
    // }
    //
    // `var` and `data` may be pinned to different devices and so we want the ops created within the
    // `Op.colocateWith(Set(data.op)) { }` block to ignore the existing stack.
    return tf.colocateWith(input, ignoreExisting = true) {
      if (input.dtype.isRefType)
        tf._refSwitch(input, predicate, name)
      else
        switch(input, predicate, name)
    }
  }
  
  fun <T : OutputLike> switch(input: T, predicate: Output, name: String = "Switch"): Array<Output> {
    return when (input) {
      is Output -> tf._switch(input, predicate, name)
      is IndexedSlices -> {
        TODO()
      }
      is SparseOutput -> {
        TODO()
      }
      else -> {
        TODO()
      }
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
   * This op handles both `Output`s and `IndexedSlices`. If inputs has a mix of
   * `Output`s and `IndexedSlices`, all inputs are converted to IndexedSlices
   * before merging.
   * @param inputs The input tensors, at most one of which is available.
   * @param name A name for this operation (optional).
   * @return A tuple containing the chosen input tensor and its index in `inputs`.
   */
  fun <T : OutputLike> merge(inputs: List<T>, name: String = "Merge"): Array<Output> {
    return when {
      inputs.all { it is Output } -> {
        inputs as List<Output>
        if (inputs.all { it.dtype.isRefType })
          ref_merge(inputs, name)
        else
          tf._merge(inputs, name)
      }
      inputs.all { it is SparseOutput } -> TODO()
      else -> TODO()
    }
  }
  
  interface API {
    
    /** Creates an op that produces the content of `input` only after all ops in `dependencies` have finished executing.
     *
     * In some cases, a user may want the output of an op to be consumed externally only after some other dependencies
     * have run first. This function ensures returns `input`, but only after all ops in `dependencies` have run. Note
     * that this means that there is no guarantee that `input` will be evaluated after any `dependencies` have run.
     *
     * @group ControlFlowOps
     * @param  dependencies Set of ops to be executed before `input`.
     * @param  input        Op output to be computed after all ops in `dependencies` have finished executing.
     * @param  name         Name for the created op (used mainly as a name scope).
     * @return Created op output.
     * @see "tensorflow.python.ops.control_flow_ops.with_dependencies"
     */
    fun withDependencies(dependencies: Set<Op>,
                         input: Output,
                         name: String = "control_dependency"): Output =
        tf.nameScope(name, dependencies + input.op!!) {
          tf.colocateWith(input) {
            tf.controlDependencies(dependencies) {
              tf.identity(input, name = tf.currentNameScope)
            }
          }
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
        return tf.device(dev) {
          groupControlDeps(deps, name)
        }
      }
      val all_deps = mutableListOf<Op>()
      return tf.nameScope(name) {
        for ((dev, deps) in ops_on_device) {
          tf.device(dev) {
            all_deps += groupControlDeps(deps)
          }
        }
        groupControlDeps(all_deps, tf.currentNameScope)
      }
    }
    
    /**
     * Return `true_fn()` if the predicate `predicate` is true else `false_fn()`.
     *
     * `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
     * `false_fn` must have the same non-zero number and type of outputs.
     
     * Note that the conditional execution applies only to the operations defined in
     * `true_fn` and `false_fn`. Consider the following simple program:
     * @param pred  A scalar determining whether to return the result of `true_fn` or `false_fn`.
     * @param true_fn The callable to be performed if predicate is true.
     * @param false_fn he callable to be performed if predicate is false.
     * @return  Tensors returned by the call to either `true_fn` or `false_fn`. If the
     * callables return a singleton list, the element is extracted from the list.
     *
     * @see "tensorflow.python.ops.control_flow_ops.cond"
     */
    fun <T> cond(pred: Output,
                 true_fn: () -> T,
                 false_fn: () -> T,
                 name: String = "cond"): Output =
        tf.nameScope(name) {
          val (p_false, p_true) = switch(pred, pred)
          val pivot_true = tf._identity(p_true, name = "switch_t")
          val pivot_false = tf._identity(p_false, name = "switch_f")
          val pred_id = tf._identity(pred, name = "pred_id")
          //Disable the fetching of tensors that are only on one branch of cond.
          for (tensor in a(p_true, p_false, pivot_true, pivot_false, pred_id))
            tensor.op!!.graph.preventFetching(tensor.op)
          
          //Build the graph for the true branch in a new context.
          val contextTrue = CondContext(pred_id, pivot_true, branch = 1)
          contextTrue.enter()
          val (originalResultTrue, resultTrue) = contextTrue.buildCondBranch(true_fn)
          contextTrue.exitResult(resultTrue)
          contextTrue.exit()
          
          // Build the graph for the false branch in a new context.
          val contextFalse = CondContext(pred_id, pivot_false, branch = 0)
          contextFalse.enter()
          val (_, resultFalse) = contextFalse.buildCondBranch(false_fn)
          contextFalse.exitResult(resultFalse)
          contextFalse.exit()
          
          // Check that the return values of the two branches have matching data types.
          resultTrue.zip(resultFalse).forEach {
            if (it.first.dtype != it.second.dtype)
              throw InvalidDataTypeException(
                  "The outputs of `trueFn` (dataType = ${it.first.dtype}) and " +
                      "`falseFn` (dataType = ${it.second.dtype}) must have the same data type.")
          }
          
          val merges = resultFalse.zip(resultTrue).map { merge(listOf(it.first, it.second))[0] }
          
          // Add to collections.
          tf.currentGraph.addToCollection(contextTrue, CondContext.Companion.COND_CONTEXT)
          tf.currentGraph.addToCollection(contextFalse, CondContext.Companion.COND_CONTEXT)
          
          unfl
          when (originalResultTrue) {
            is Op -> merges.first().op
            is Output ->
          }
          
          TODO()
//          val res_t = tf.condContext(pred_id, pivot_true, branch = 1) {
//            it.buildCondBranch(true_fn)
//          }
//          val res_f = tf.condContext(pred_id, pivot_false, branch = 0) {
//            it.buildCondBranch(false_fn)
//          }
//    val res_t = buildCondBranch(predicate, pivot_1, 1, true_fn)
//     = buildCondBranch(predicate, pivot_2, 0, false_fn)
//          merge(res_t, res_f)[0]
        }
    
  }
}