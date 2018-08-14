package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.ops
import wumo.sim.tensorflow.tf
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
  /** Control flow context containing this context. */
  val outerContext = tf.currentControlFlowContext
  /**Set of values that have already been seen in this context.*/
  val values = hashSetOf<String>()
  /**Set of values referenced by but external to this context.*/
  val external_values = hashMapOf<String, Output>()
  /** Contains the stack of control flow contexts that have been entered so far. */
  val contextStack = ArrayDeque<ControlFlowContext?>()
  
  /**
   * Returns the first ancestor [CondContext] containing this context.
   */
  fun condContext(): CondContext? {
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
   */
  fun whileContext(stopContext: ControlFlowContext? = null): WhileContext? {
    var context: ControlFlowContext? = this
    while (context != null) {
      if (context is WhileContext || context === stopContext)
        return context as WhileContext
      context = context.outerContext
    }
    return null
  }
  
  /**Returns the first ancestor XLAContext containing this context.*/
  open fun xlaContext(): XLAControlFlowContext? = outerContext?.xlaContext()
  
  abstract fun addOp(op: Op)
  
  abstract val gradState: GradientLoopState?
}

abstract class XLAControlFlowContext : ControlFlowContext() {
  override fun xlaContext(): XLAControlFlowContext? = this
}