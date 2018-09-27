package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.basic.times
import wumo.sim.tensorflow.ops.gen.gen_control_flow_ops
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf
import wumo.sim.util.NONE
import wumo.sim.util.a
import wumo.sim.util.groupBy
import kotlin.collections.component1
import kotlin.collections.component2

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

object control_flow_ops {
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
          val context = getOutputContext(i.op)
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
    var op = value.op
    while (op.opType in identityOpTypes)
      op = op.inputs[0].op
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
          whileContext.gradLoopState != null &&
              isContainingContext(whileContext.gradLoopState.forwardContext,
                                  inputWhileContext) -> null
          // `op` is in a gradient context and `inputOp` is in the associated forward pass context or an ancestor
          // thereof. This case is needed to build while loop gradients. Note that we theoretically also need this
          // case for custom gradient functions that close over tensors from ancestor contexts, but this has not been
          // verified yet.
          whileContext.gradLoopState != null &&
              whileContext.gradLoopState.forwardContext === inputWhileContext?.outerContext -> null
          // `op` is in a gradient context and `inputOp` is in a child of the associated forward pass context. This
          // case is needed for the gradients of while loops with conditionals.
          inputWhileContext?.gradLoopState != null &&
              inputWhileContext.gradLoopState.forwardContext === whileContext -> null
          // `inputOp` is in the gradient context of `op`'s context. This case is needed when the gradient of a while
          // loop gradient is requested (this will eventually fail unless there is a `stopGradient` op or similar).
          inputWhileContext?.gradLoopState != null &&
              inputContext.gradLoopState?.forwardContext?.gradLoopState != null &&
              inputContext.gradLoopState!!.forwardContext.gradLoopState!!.forwardContext === whileContext -> null
          // `inputOp` is in the gradient gradient context of `op`'s context. This case is needed when the gradient of
          // a while loop gradient is requested (this will eventually fail unless there is a `stopGradient` op or
          // similar).
          else -> "Cannot use '${inputOp.name}' as input to '${op.name}' because they are in different while loops."
        }
      }
    }
    if (errorMsg != null) throw InvalidArgumentException(errorMsg)
  }
  
  /** Calculates a maximum size for use by stack ops inside XLA while loops.
   *
   * @param  value            Value inside the while loop forward context. Used for printing error messages.
   * @param  whileLoopContext Forward context inside which value resides. This does not always match the value's
   *                          immediate context, as `value` may be inside e.g., a cond context, inside the while loop.
   * @return Tensor containing the `maxSize` to feed to a stack initializer.
   * @throws InvalidArgumentException If `value` is nested inside a while loop that either lacks a `maximumIterations`
   *                                  parameter, or whose `maximumIterations` parameter is inside a while loop that is
   *                                  a parent of the calling context, and cannot be evaluated at graph build time
   *                                  (i.e., statically) to a constant value.
   */
  internal fun getMaxSizeFromNestedMaximumIterations(
      value: Output,
      whileLoopContext: WhileContext?
  ): Output {
    val valueName = value.name
    // `currentContext` is the context that `tf.gradients()` was called in.
    val currentContext = tf.currentControlFlowContext
    val currentContextName = currentContext?.name ?: ""
    
    // Loop through all containing while-loop contexts between the value and the current context, multiplying together
    // each context's `maxIterations`, in order to get the maximum stack size.
    var maxSize = tf.const(1)
    
    var currentWhileContext = whileLoopContext
    while (currentWhileContext != null) {
      val maxIter = currentWhileContext.maximumIterations
      if (maxIter == null)
        throw InvalidArgumentException(
            "Cannot create a gradient accumulator for tensor '$valueName', inside an XLA while loop, because " +
                "'maximumIterations' was not passed to the `tf.whileLoop()` call " +
                "('${currentWhileContext.name}').")
      else {
        val maxIterContext = maxIter.op.controlFlowContext
        // If `maxIterContext` (non-strictly) contains `currentContext`, then it is ok to use.
        if (isContainingContext(currentContext, maxIterContext))
          maxSize *= maxIter
        else {
          // We cannot use `maxIter` because it is defined in a nested while-loop or cond context, and so
          // an error will be thrown if we try to use it as input to any ops in `currentContext` (e.g., `maxSize` or
          // the final accumulator stack). We attempt to get a constant value out to use instead.
          val constMaxIter = constantValue<Any>(maxIter)
          if (constMaxIter == null)
            throw InvalidArgumentException(
                "Cannot create a gradient accumulator for tensor '$valueName', inside an XLA while loop, because " +
                    "the 'maximumIterations' tensor ('${maxIter.name}') for while-loop context " +
                    "'${currentWhileContext.name}' must be statically known (e.g., a constant value or " +
                    "known shape dimension), or must be defined at or outside the while-loop context " +
                    "'$currentContextName' (currently defined in '${maxIterContext?.name}').")
          else
            maxSize *= constMaxIter
        }
      }
      // Find the next outer while-loop context, or stop if we have reached the `tf.gradients()` context.
      currentWhileContext = getContainingWhileContext(currentWhileContext.outerContext, stopContext = currentContext)
    }
    return maxSize
  }
  
  private fun groupControlDeps(dev: String, deps: MutableSet<Op>, name: String = "noOp") =
      tf.with(device = dev, controlDependencies = deps) {
        tf.noOp(name)
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
                         name: String = "Switch"): List<Output> {
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
      if (input.dataType.isRefType)
        tf.refSwitch(input, predicate, name)
      else
        tf.switch(input, predicate, name)
    }
  }
  
  interface API {
    
    fun abort(errorMsg: String = "", exitWithoutError: Boolean = false, name: String = "Abort"): Op {
      return gen_control_flow_ops.abort(errorMsg, exitWithoutError, name)
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
                 name: String = "cond"): T =
        tf.nameScope(name) {
          val (p_false, p_true) = switch(pred, pred)
          val pivot_true = tf.identity(p_true, name = "switch_t")
          val pivot_false = tf.identity(p_false, name = "switch_f")
          val pred_id = tf.identity(pred, name = "pred_id")
          //Disable the fetching of tensors that are only on one branch of cond.
          for (tensor in a(p_true, p_false, pivot_true, pivot_false, pred_id))
            tensor.op.graph.preventFetching(tensor.op)
          
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
            if (it.first.dataType.baseDataType != it.second.dataType.baseDataType)
              throw InvalidDataTypeException(
                  "The outputs of `trueFn` (dataType = ${it.first.dataType.baseDataType}) and " +
                      "`falseFn` (dataType = ${it.second.dataType.baseDataType}) must have the same data type.")
          }
          
          val merges = resultFalse.zip(resultTrue).map { merge(listOf(it.first, it.second))[0] }
          
          // Add to collections.
          tf.currentGraph.addToCollection(contextTrue, CondContext.Companion.COND_CONTEXT)
          tf.currentGraph.addToCollection(contextFalse, CondContext.Companion.COND_CONTEXT)
          
          CondContext.unflatten(originalResultTrue, merges)
        }
    
    fun controlTrigger(name: String = "ControlTrigger"): Op {
      return gen_control_flow_ops.controlTrigger(name)
    }
    
    /** Creates an op that creates or finds a child frame, and makes `input` available to that child frame.
     *
     * The op is used together with `exit` to create loops in the graph. The unique `frameName` is used by the `Executor`
     * to identify frames. If `isConstant` is `true`, then the output is a constant in the child frame. Otherwise, it may
     * be changed in the child frame. At most `parallelIterations` iterations are run in parallel in the child frame.
     *
     * @param  input              Tensor to be made available to the child frame.
     * @param  frameName          Name of the child frame.
     * @param  isConstant         If `true`, the output is constant within the child frame.
     * @param  parallelIterations Number of iterations allowed to run in parallel.
     * @param               useRef: Boolean = true,
    : If true, use ref_enter if data is of ref type.
     * @param  useInputShape      If `true`, the output tensor's shape is manually set to the input tensor's shape.
     * @param  name               Name for the created op.
     * @return Created op output, which is the same as `input`.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T : OutputLike> enter(input: T, frameName: String, isContant: Boolean = false, parallelIterations: Int = 10,
                               useRef: Boolean = true, useInputShape: Boolean = true, name: String = "Enter"): T =
        when (input) {
          is Output -> {
            val result = if (input.dataType.isRefType && useRef)
              gen_control_flow_ops.refEnter(input, frameName, isContant, parallelIterations.toLong(), name)
            else
              gen_control_flow_ops.enter(input, frameName, isContant, parallelIterations.toLong(), name)
            if (useInputShape)
              result.setShape(input.shape)
            result
          }
          is IndexedSlices -> {
            val values = enter(input.values, frameName, isContant, parallelIterations, useRef, useInputShape, name)
            val indices = enter(input.indices, frameName, isContant, parallelIterations, useRef, useInputShape, "indices")
            val denseShape = if (input.denseShape != null)
              enter(input.denseShape, frameName, isContant, parallelIterations, useRef, useInputShape, "dense_shape")
            else null
            IndexedSlices(indices, values, denseShape)
          }
          is SparseOutput -> {
            val values = enter(input.values, frameName, isContant, parallelIterations, useRef, useInputShape, name)
            val indices = enter(input.indices, frameName, isContant, parallelIterations, useRef, useInputShape, "indices")
            val denseShape = enter(input.denseShape!!, frameName, isContant, parallelIterations, useRef, useInputShape, "dense_shape")
            SparseOutput(indices, values, denseShape)
          }
          else -> NONE()
        } as T
    
    /** Creates an op that exits from the current frame to its parent frame.
     *
     * The op makes `input` available to the parent frame.
     *
     * @param  input Tensor to be made available to the parent frame.
     * @param  name  Name for the created op.
     * @return Created op output, which is the same as `input`.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T : OutputLike> exit(input: T, name: String = "Exit"): T =
        when (input) {
          is Output -> if (input.dataType.isRefType)
            gen_control_flow_ops.refExit(input, name)
          else
            gen_control_flow_ops.exit(input, name)
          is IndexedSlices -> {
            val values = exit(input.values, name)
            val indices = exit(input.indices, "indices")
            val denseShape = if (input.denseShape != null)
              exit(input.denseShape, name)
            else null
            IndexedSlices(indices, values, denseShape)
          }
          is SparseOutput -> {
            val values = exit(input.values, name)
            val indices = exit(input.indices, "indices")
            val denseShape = exit(input.denseShape!!, name)
            SparseOutput(indices, values, denseShape)
          }
          else -> NONE()
        } as T
    
    /**
     * Create an op that groups multiple operations.
     *
     * When this op finishes, all ops in [inputs] have finished. This op has no
     * output.
     *
     * @param inputs Zero or more tensors to group.
     * @param name A name for this operation (optional).
     *
     * @return An Operation that executes all its inputs.
     */
    fun group(inputs: Iterable<Op>, name: String = "group_deps"): Op {
      if (inputs.none())
        return tf.noOp(name)
      // Sorts *inputs according to their devices.
      val ops_on_device = inputs.groupBy { it.device }
      if (ops_on_device.size == 1) {
        // 1-level tree. The root node is the returned NoOp node.
        val (dev, deps) = ops_on_device.entries.first()
        return groupControlDeps(dev, deps, name)
      }
      // 2-level tree. The root node is the returned no-op node. `dependencies` contains 1 NoOp node for each device.
      val deps = ops_on_device.asSequence()
          .sortedBy { it.key }
          .mapTo(mutableSetOf()) { (dev, ops) ->
            groupControlDeps(dev, ops)
          }
      return tf.controlDependencies(deps) {
        tf.noOp(name)
      }
    }
    
    fun loopCond(input: Output, name: String = "LoopCond"): Output {
      return gen_control_flow_ops.loopCond(input, name)
    }
    
    fun nextIteration(data: Output, name: String = "NextIteration"): Output {
      return gen_control_flow_ops.nextIteration(data, name)
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
    @Suppress("UNCHECKED_CAST")
    fun <T : OutputLike> merge(inputs: List<T>, name: String = "Merge"): List<Output> {
      return when {
        inputs.all { it is Output } -> {
          inputs as List<Output>
          if (inputs.all { it.dataType.isRefType })
            gen_control_flow_ops.refMerge(inputs, name)
          else
            gen_control_flow_ops.merge(inputs, name)
        }
        inputs.all { it is SparseOutput } -> TODO()
        else -> TODO()
      }
    }
    
    /** Creates an op that makes its input available to the next iteration.
     *
     * @param  input Tensor to make available to the next iteration.
     * @param  name  Name for the created op.
     * @return Created op output, which is the same as `input`.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T : OutputLike> nextIteration(input: T, name: String = "NextIteration"): T =
        when (input) {
          is Output ->
            if (input.dataType.isRefType)
              gen_control_flow_ops.refNextIteration(input, name)
            else
              gen_control_flow_ops.nextIteration(input, name)
          is IndexedSlices -> {
            val values = nextIteration(input.values, name)
            val indices = nextIteration(input.indices, "indices")
            val denseShape = if (input.denseShape != null)
              nextIteration(input.denseShape, "dense_shape")
            else null
            IndexedSlices(indices, values, denseShape)
          }
          is SparseOutput -> {
            val values = nextIteration(input.values, name)
            val indices = nextIteration(input.indices, "indices")
            val denseShape = nextIteration(input.denseShape!!, "dense_shape")
            SparseOutput(indices, values, denseShape)
          }
          else -> NONE()
        } as T
    
    fun noOp(name: String = "NoOp"): Op {
      return gen_control_flow_ops.noOp(name)
    }
    
    fun refEnter(data: Output, frameName: String, isConstant: Boolean = false, parallelIterations: Long = 10L, name: String = "RefEnter"): Output {
      return gen_control_flow_ops.refEnter(data, frameName, isConstant, parallelIterations, name)
    }
    
    fun refExit(data: Output, name: String = "RefExit"): Output {
      return gen_control_flow_ops.refExit(data, name)
    }
    
    fun refMerge(inputs: List<Output>, name: String = "RefMerge"): List<Output> {
      return gen_control_flow_ops.refMerge(inputs, name)
    }
    
    fun refNextIteration(data: Output, name: String = "RefNextIteration"): Output {
      return gen_control_flow_ops.refNextIteration(data, name)
    }
    
    fun refSelect(index: Output, inputs: List<Output>, name: String = "RefSelect"): Output {
      return gen_control_flow_ops.refSelect(index, inputs, name)
    }
    
    fun refSwitch(data: Output, pred: Output, name: String = "RefSwitch"): List<Output> {
      return gen_control_flow_ops.refSwitch(data, pred, name)
    }
    
    fun <T : OutputLike> switch(input: T, predicate: Output, name: String = "Switch"): List<Output> {
      val _input = input as OutputLike
      return when (_input) {
        is Output -> gen_control_flow_ops.switch(_input, predicate, name)
        is IndexedSlices -> {
          TODO()
        }
        is SparseOutput -> {
          TODO()
        }
      }
    }
    
    /** Group tensors together.
     * This creates a tuple of tensors with the same values as the `tensors`
     * argument, except that the value of each tensor is only returned after the
     * values of all tensors have been computed.
     *
     * `control_inputs` contains additional ops that have to finish before this op
     * finishes, but whose outputs are not returned.
     *
     * This can be used as a "join" mechanism for parallel computations: all the
     * argument tensors can be computed in parallel, but the values of any tensor
     * returned by `tuple` are only available after all the parallel computations
     * are done.
     *
     * See also @{tf.group$group} and
     * @{tf.control_dependencies$control_dependencies}.
     *
     * @group ControlFlowOps
     * @param  inputs        Op outputs being grouped.
     * @param  controlInputs Set of additional ops that have to finish before this op finishes, but whose outputs are not
     *                       returned.
     * @param  name          Name for the created ops (used mainly as a name native).
     * @return Created op outputs, which in this case are the values of `inputs`.
     */
    fun tuple(inputs: List<OutputLike?>, controlInputs: Set<Op> = emptySet(), name: String = "tuple"): List<OutputLike?> {
      val gatingOps = inputs.asSequence().filterNotNull().mapTo(mutableSetOf()) { it.op }
      return if (gatingOps.isEmpty())
        inputs
      else
        tf.nameScope(name, gatingOps) {
          val gate = group(gatingOps + controlInputs)
          inputs.map {
            if (it == null)
              it
            else
              withDependencies(setOf(gate), it)
          }
        }
    }
    
    /** Creates an op that produces the content of `input` only after all ops in `dependencies` have finished executing.
     *
     * In some cases, a user may want the output of an op to be consumed externally only after some other dependencies
     * have run first. This function ensures returns `input`, but only after all ops in `dependencies` have run. Note
     * that this means that there is no guarantee that `input` will be evaluated after any `dependencies` have run.
     *
     * @group ControlFlowOps
     * @param  dependencies Set of ops to be executed before `input`.
     * @param  input        Op output to be computed after all ops in `dependencies` have finished executing.
     * @param  name         Name for the created op (used mainly as a name native).
     * @return Created op output.
     * @see "tensorflow.python.ops.control_flow_ops.with_dependencies"
     */
    fun withDependencies(dependencies: Set<Op>,
                         input: OutputLike,
                         name: String = "control_dependency"): Output =
        tf.nameScope(name, dependencies + input.op) {
          tf.colocateWith(input.op) {
            tf.controlDependencies(dependencies.toMutableSet()) {
              tf.identity(input, name = tf.currentNameScope)
            }
          }
        }
  }
}