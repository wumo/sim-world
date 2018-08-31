package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.isSwitch
import wumo.sim.tensorflow.ops.ops.graphConstructionScope
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.RESOURCE
import java.util.*

/**
 * The base class for control flow context.
 
 *  The usage pattern is a sequence of (Enter, Exit) followed by a final
 * ExitResult.
 
 * We maintain the following state for control flow contexts during graph
 * construction:
 * 1. graph has _control_flow_context: the current context used to
 * construct new nodes. Changed by ctxt.Enter() and ctxt.Exit()
 * 2. op has [Op.controlFlowContext]: the context to which the op belongs.
 * Set at the time the op is created. Immutable.
 * 3. A [ControlFlowContext] has [outerContext]: the context in which this
 * context is created. Set at the time a context is created. Immutable.
 * 4. A ControlFlowContext has _context_stack.
 * Pushed and popped by ctxt.Enter() and ctxt.Exit()
 */
abstract class ControlFlowContext {
  
  /** Name of this control flow context. */
  abstract val name: String
  
  /** Control flow context containing this context. */
  val outerContext = tf.currentControlFlowContext
  /**Set of values that have already been seen in this context.*/
  val values = hashSetOf<String>()
  /**Set of values referenced by but external to this context.*/
  val externalValues = hashMapOf<String, Output>()
  /** Contains the stack of control flow contexts that have been entered so far. */
  val contextStack = ArrayDeque<ControlFlowContext?>()
  /** Returns the control pivot op output for this context, or null.*/
  open val controlPivot: Op? = null
  
  /**
   * Returns the first ancestor [CondContext] containing this context.
   * @see "tensorflow.python.ops.control_flow_util.GetContainingCondContext"
   */
  open val condContext: CondContext?
    get() {
      var context: ControlFlowContext? = this
      while (context != null) {
        if (context is CondContext) return context
        context = context.outerContext
      }
      return null
    }
  
  /**
   * Returns the first ancestor [WhileContext] containing this context.
   
   * @param stopContext If provided, the search will end if it sees [stopContext].
   * @see "tensorflow.python.ops.control_flow_util.GetContainingWhileContext"
   */
  open fun whileContext(stopContext: ControlFlowContext? = null): WhileContext? {
    var context: ControlFlowContext? = this
    while (context != null) {
      if (context is WhileContext || context === stopContext)
        return context as WhileContext
      context = context.outerContext
    }
    return null
  }
  
  /**Returns the first ancestor [XLAControlFlowContext] containing this context.
   * @see "tensorflow.python.ops.control_flow_util.GetContainingXLAContext"
   */
  open fun xlaContext(): XLAControlFlowContext? = outerContext?.xlaContext()
  
  /**
   * Adds `output` to the current context and its outer context recursively.
   * @see "tensorflow.python.ops.control_flow_ops.CondContext#AddValue"
   * @see "tensorflow.python.ops.control_flow_ops.WhileContext#AddValue"
   */
  abstract fun addValue(output: Output): Output
  
  /** Adds [op] to the current context.
   * @see "tensorflow.python.ops.control_flow_ops.CondContext#AddOp"
   * @see "tensorflow.python.ops.control_flow_ops.WhileContext#AddOp"
   */
  open fun addOp(op: Op) = addInternal(op)
  
  /** Adds [op] to the current context. We move any external control dependencies of the op to the control flow pivot,
   * to ensure they get executed. */
  abstract fun addInternal(op: Op)
  
  /** Returns `true` if back-propagation is supported for this control flow context.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#back_prop"
   */
  abstract val backPropagate: Boolean
  
  /**
   * Gradient loop state for this context, used for back-propagation.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#grad_state"
   */
  abstract val gradLoopState: GradientLoopState?
  
  /** Enters this control flow context.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#Enter"
   */
  fun enter() {
    graphConstructionScope.value.controlFlowContext?.let { contextStack.addLast(it) }
    graphConstructionScope.value = graphConstructionScope.value.copy(
        controlFlowContext = this,
        outerContext = graphConstructionScope.value)
  }
  
  /** Exits this control flow context.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#Exit"
   */
  fun exit() {
    graphConstructionScope.value = graphConstructionScope.value.copy(
        controlFlowContext = contextStack.pollLast(),
        outerContext = graphConstructionScope.value)
  }
  
  /** Makes a sequence of tensors available in the outer context.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#ExitResult"
   */
  fun exitResult(result: List<OutputLike>) {
    outerContext?.let { c ->
      result.forEach {
        c.values += it.name
      }
    }
  }
  
  /**
   * Enters a control flow context for building a gradient colocated with `colocationOps`.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#EnterGradientColocation"
   */
  fun enterGradientColocation(colocationOps: Set<Op>, gradientUID: String) {
    outerContext?.enterGradientColocation(colocationOps, gradientUID)
  }
  
  /** Exits a control flow context for building a gradient colocated with `colocationOps`.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#ExitGradientColocation"
   */
  fun exitGradientColocation(colocationOps: Set<Op>, gradientUID: String) {
    outerContext?.exitGradientColocation(colocationOps, gradientUID)
  }
  
  /** Removes any external control dependency on this op and returns the remaining internal control inputs and any
   * external control inputs that were removed.
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowContext#_RemoveExternalControlEdges"
   */
  internal fun removeExternalControlEdges(op: Op): Pair<Set<Op>, Set<Op>> {
    // A control input of 'op' is internal if it is in the same while loop context as the enclosing while loop context
    // of this context.
    val whileCtx = whileContext()
    val internalControlInputs = when (whileCtx) {
      null -> op.controlInputs
      else -> op.controlInputs.filterTo(mutableSetOf()) { control_flow_ops.getOutputContext(it)?.whileContext() == whileCtx }
    }
    val externalControlInputs = if (internalControlInputs.size != op.controlInputs.size) {
      val externalControlInputs = op.controlInputs - internalControlInputs
      op.removeAllControlInputs()
      op.addControlInputs(internalControlInputs)
      externalControlInputs
    } else
      emptySet<Op>()
    return internalControlInputs to externalControlInputs
  }
  
  companion object {
    /** Create a `zerosLike` op for the specified op output, while taking into account control flow contexts.
     * @see "tensorflow.python.ops.control_flow_ops.ZerosLikeOutsideLoop"
     */
    internal fun zerosLikeOutsideLoop(op: Op, index: Int): Output {
      val value = op.outputs[index]
      return if (isSwitch(op)) {
        op.controlFlowContext?.let {
          it as CondContext
          val switch = control_flow_ops.switch(op.inputs[0], it.predicate)[1 - it.branch]
          // We are in a conditional context and so we use a switch to create zeros only when needed.
          if (value.dataType == RESOURCE)
            tf.controlDependencies(mutableSetOf(switch.op)) {
              tf.zeros(tf.variableShape(switch))
            }
          else {
            val zerosShape = tf.shape(switch, optimize = false)
            // Ensure ops created within array_ops.zeros are dominated by switch in
            // cond context.
            tf.controlDependencies(mutableSetOf(switch.op)) {
              tf.zeros(zerosShape, value.dataType)
            }
          }
        } ?: tf.zerosLike(value, optimize = false)
      } else {
        if (value.dataType == RESOURCE)
          tf.zeros(tf.variableShape(value))
        else
          tf.zerosLike(value, optimize = false)
      }
    }
  }
}

abstract class XLAControlFlowContext : ControlFlowContext() {
  override fun xlaContext(): XLAControlFlowContext? = this
}