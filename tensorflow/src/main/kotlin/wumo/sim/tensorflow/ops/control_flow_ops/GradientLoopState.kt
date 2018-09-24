package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.core.InvalidArgumentException
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.getLoopConstantEnter
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.getOutputContext
import wumo.sim.tensorflow.tensor.constantValue
import wumo.sim.tensorflow.tf

/** State used for constructing the gradient graph for a while loop.
 *
 * We create a [[GradientLoopState]] for each while loop in the forward pass and its corresponding while loop in the
 * backward pass. This gives us access to both the forward and the backward [[WhileLoopContext]]s.
 *
 * During the construction of the gradient graph, whenever we detect a forward value that is needed for the backward
 * pass, we create a history accumulator and add it to `historyMap`. Whenever we back-propagate a loop switch op, we
 * add the corresponding gradient merge op in `switchMap`.
 *
 * @param  forwardContext         While-loop context used for the forward pass.
 * @param  outerGradState The gradient loop state used for the outer loop.
 *
 */
class GradientLoopState(val forwardContext: WhileContext, val outerGradState: GradientLoopState?) {
  
  /** Map that records all tensors needed for back-propagation. */
  internal val historyMap: MutableMap<String, Output> = mutableMapOf()
  
  /** Map that records all the switch ops needed for back-propagation. */
  internal val switchMap: MutableMap<Op, OutputLike> = mutableMapOf()
  
  /** List containing all "unused" exits. */
  internal val unusedExits: MutableSet<Output> = mutableSetOf()
  
  /** List containing all "deferred" exits. */
  internal val deferredExits: MutableSet<Output> = mutableSetOf()
  
  /** List containing all forward loop exits. */
  internal val forwardLoopExits: MutableSet<Output> = HashSet(forwardContext.loopExits)
  
  /** Number of exits we expect to see during the backward pass, but have not seen yet. */
  internal var pendingExitsCount: Int = forwardContext.loopExits.size
  
  /**Value of the loop counter for the next iteration added by `addForwardLoopCounter()`.*/
  val forwardIndex: Output
  /**Value of the loop counter for the current iteration added by `addBackwardLoopCounter()`.*/
  val backwardIndex: Output
  /**While loop context used for backpropagation.*/
  val backwardContext: WhileContext
  
  init {
    val outerForwardContext = outerGradState?.forwardContext ?: forwardContext.outerContext
    // Add the forward loop counter.
    outerForwardContext?.enter()
    val (count, forwardIndex) = forwardContext.addForwardLoopCounter(outerGradState)
    outerForwardContext?.exit()
    this.forwardIndex = forwardIndex
    
    // Add the backward while-loop context, and the backward loop counter.
    if (outerGradState != null) {
      // This is a nested loop. Remember the iteration counts for each execution of this inner loop.
      outerForwardContext?.values?.add(count.name)
      val historyCount = outerGradState.addForwardAccumulator(count)
      outerGradState.backwardContext.enter()
      backwardContext = WhileContext(forwardContext.maximumIterations,
                                     forwardContext.parallelIterations,
                                     forwardContext.backPropagate,
                                     forwardContext.swapMemory, this,
                                     forwardContext.name)
      val realCount = outerGradState.addBackwardAccumulatedValue(historyCount, count)
      backwardIndex = backwardContext.addBackwardLoopCounter(realCount, outerGradState)
      outerGradState.backwardContext.exit()
    } else {
      outerForwardContext?.enter()
      backwardContext = WhileContext(forwardContext.maximumIterations,
                                     forwardContext.parallelIterations,
                                     forwardContext.backPropagate,
                                     forwardContext.swapMemory,
                                     this,
                                     forwardContext.name)
      backwardIndex = backwardContext.addBackwardLoopCounter(count, outerGradState)
      outerForwardContext?.exit()
    }
  }
  
  /** Control trigger node for synchronization in the backward loop. One main use is to keep the pop ops of a stack
   * executed in the iteration order. */
  internal val forwardSync: Op by lazy {
    val syncOp = tf.controlDependencies(mutableSetOf()) {
      tf.controlTrigger("f_sync")
    }
    syncOp.controlFlowContext = forwardContext
    forwardIndex.op.addControlInput(syncOp)
    syncOp
  }
  
  /** Control trigger node for synchronization in the backward loop. One main use is to keep the pop ops of a stack
   * executed in the iteration order. */
  internal val backwardSync: Op by lazy {
    val syncOp = tf.controlDependencies(mutableSetOf()) {
      tf.controlTrigger("b_sync")
    }
    syncOp.controlFlowContext = backwardContext
    backwardIndex.op.addControlInput(syncOp)
    syncOp
  }
  
  /** Gets the real value of `value`.
   *
   * If back-propagation "uses" a value produced by the forward loop, an accumulator is added in the forward loop to
   * collect its values. We use the accumulated value. This method must be called for the backward loop context.
   * `value` must be in the forward loop and is needed for back-propagation.
   * @see "tensorflow.python.ops.control_flow_ops.GradLoopState#GetRealValue"
   */
  internal fun getRealValue(value: Output): Output =
      historyMap.getOrPut(value.name) {
        var realValue: Output? = null
        var historyValue: Output? = null
        var currentValue = value
        var currentGradientLoopState = this
        outer@ while (true) {
          val enterOp = getLoopConstantEnter(currentValue)
          when {
            enterOp != null -> {
              // Special case: `currentValue` comes from a constant enter node.
              currentValue = enterOp.inputs[0]
              if (currentGradientLoopState.outerGradState != null) {
                currentGradientLoopState = currentGradientLoopState.outerGradState!!
              } else {
                // We are now outside all nested loops for this gradient and so `value` is a loop invariant and there is
                // no need to save its history. We just make `currentValue` enter the right control flow context.
                realValue = backwardContext.addValue(currentValue)
                break@outer
              }
            }
            currentValue.op.opType == "Const" -> {
              // If the value to be forwarded is a constant, clone the constant in
              // the gradient loop rather than using a stack.
              // TODO(phawkins): consider hoisting the constant out of the loop
              // instead.
              realValue = tf.const(constantValue<Any>(currentValue)!!)
              break@outer
            }
            else -> {
              // Record the history of this value in forward_ctxt.
              backwardContext.exit()
              historyValue = currentGradientLoopState.addForwardAccumulator(currentValue)
              backwardContext.enter()
              break@outer
            }
          }
        }
        if (realValue == null) {
          // Add the stack pop op in the backward context.
          realValue = currentGradientLoopState.addBackwardAccumulatedValue(historyValue!!, currentValue)
          if (currentGradientLoopState != this)
            realValue = backwardContext.addValue(realValue)
        }
        realValue
      }
  
  /** Adds an accumulator for each forward tensor that is needed in the backward loop.
   *
   * This is added to the forward loop the first time when a tensor is used by the back-propagation gradient
   * computation loop. We create an accumulator that collects the value of the tensor at each iteration.
   *
   * The pseudocode is: `acc = newStack(); while (pivot) { acc = stackPush(acc, value); }`
   *
   * We make sure that the stack push op in one iteration is executed before next iteration. This is achieved by adding
   * a control edge from `forwardIndex.op.inputs(0).op` to the push op, and another control edge from the push op to
   * either `forwardIndex.op` or `forwardSync`.
   *
   * @param  value      Source tensor in the forward loop that is to be accumulated.
   * @param  deadBranch Set to `true`, if and only if `value` is on a dead branch of a conditional.
   * @return Resource handle to a stack that contains the accumulated history of the tensor.
   */
  internal fun addForwardAccumulator(value: Output, deadBranch: Boolean = false): Output {
    // `currentContext` is the context that `tf.gradients()` was called in.
    val currentContext = tf.currentControlFlowContext
    return tf.controlDependencies(mutableSetOf()) {
      currentContext?.enter()
      val accumulator = tf.colocateWith(mutableSetOf(value.op)) {
        // We only need to pass `maximumIterations` to the stack if we are inside an XLA context.
        val maxSize =
            if (value.op.isInXLAContext)
              control_flow_ops.getMaxSizeFromNestedMaximumIterations(value, forwardContext)
            else
              tf.const(-1)
        tf.stackV2(maxSize, value.dataType, name = "ForwardAccumulator")
      }
      currentContext?.exit()
      // Make the `accumulator` available in the forward context.
      val enterAccumulator = forwardContext.addValue(accumulator)
      // Add the stack push op in the context of `value.op`.
      val valueContext = getOutputContext(value.op)
      val stackPushOp = when (valueContext) {
        forwardContext -> {
          // `value` is not nested in the forward context.
          forwardContext.enter()
          val stackPushOp = tf.stackPushV2(enterAccumulator, value, forwardContext.swapMemory).op
          forwardContext.exit()
          // Protect the stack push and order it before `forwardIndex`.
          forwardIndex.op.addControlInput(stackPushOp)
          stackPushOp
        }
        is CondContext -> {
          // `value` is in a conditional context within the forward context.
          val stackPushOp =
              if (deadBranch) {
                // Special case for creating a zero tensor for a dead branch of a switch.
                valueContext.outerContext?.enter()
                val stackPushOp = tf.stackPushV2(enterAccumulator, value, forwardContext.swapMemory).op
                valueContext.outerContext?.exit()
                stackPushOp.controlFlowContext = valueContext
                stackPushOp
              } else {
                valueContext.enter()
                val stackPushOp = tf.stackPushV2(enterAccumulator, value, forwardContext.swapMemory).op
                valueContext.exit()
                stackPushOp
              }
          // Protect the stack push and order it before `forwardSync`.
          forwardSync.addControlInput(stackPushOp)
          stackPushOp
        }
        else -> throw InvalidArgumentException("'valueContext' is not a CondContext: $valueContext.")
      }
      // Order the stack push after the successor of `forwardIndex`.
      stackPushOp.addControlInput(forwardIndex.op.inputs[0].op)
      accumulator
    }
  }
  
  /** Adds the getter for an accumulated value in the backward context.
   *
   * This is added to the back-propagation loop. It is called in the backward context to get the value of an
   * accumulated value. The stack pop op must be guarded by the predicate of the controlling conditional context.
   *
   * @param  historyValue Resource handle to stack containing the "history" of a value.
   * @param  value        Value that is pushed into the stack.
   * @param  deadBranch   Set to `true`, if and only if `value` is on a dead branch of a conditional.
   * @return Current value (popped from the top of the stack).
   */
  internal fun addBackwardAccumulatedValue(
      historyValue: Output, value: Output, deadBranch: Boolean = false): Output {
    val historyContext = historyValue.op.controlFlowContext
    // Find the cond context that controls `historyValue`, if any.
    var condContext: CondContext? = null
    var valueContext = value.op.controlFlowContext
    while (valueContext != null && valueContext != historyContext) {
      if (valueContext is CondContext) {
        condContext = valueContext
        break
      }
      valueContext = valueContext.outerContext
    }
    val stackPopOp = tf.controlDependencies(mutableSetOf()) {
      backwardContext.enter()
      val stackHandle = condContext?.let {
        // Guard the stack pop op with a switch, if it is controlled by a conditional.
        var predicate: Output? = null
        var gradientLoopState: GradientLoopState? = this
        while (predicate == null && gradientLoopState != null) {
          predicate = gradientLoopState.historyMap[condContext.predicate.name]
          gradientLoopState = gradientLoopState.outerGradState
        }
        if (predicate == null)
          predicate = condContext.predicate
        val branch = if (deadBranch) 1 - condContext.branch else condContext.branch
        control_flow_ops._switchRefOrTensor(historyValue, predicate)[branch]
      } ?: historyValue
      val stackPopOp = tf.stackPopV2(stackHandle, value.dataType)
      stackPopOp.setShape(value.shape)
      backwardContext.exit()
      stackPopOp
    }
    if (backwardContext.parallelIterations > 1) {
      // All stack pop ops are ordered after `pivotForBody` and before `backwardSync`.
      backwardSync.addControlInput(stackPopOp.op)
    }
    return stackPopOp
  }
}