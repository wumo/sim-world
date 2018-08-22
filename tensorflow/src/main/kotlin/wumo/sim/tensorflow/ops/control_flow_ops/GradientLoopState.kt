package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.OutputLike

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
class GradientLoopState(val forwardContext: WhileContext, val outerGradState: GradientLoopState) {
  
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
  
  val forwardIndex: Output
  val backwardIndex: Output
  val backwardContext: WhileContext
  
  init {
    val outerForwardContext = outerGradState?.forwardContext ?: forwardContext.outerContext
    // Add the forward loop counter.
    outerForwardContext?.enter()
    val (count, forwardIndex) = forwardContext.addForwardLoopCounter(outerGradState)
    
  }
  
  /** Gets the real value of `value`.
   *
   * If back-propagation "uses" a value produced by the forward loop, an accumulator is added in the forward loop to
   * collect its values. We use the accumulated value. This method must be called for the backward loop context.
   * `value` must be in the forward loop and is needed for back-propagation.
   * @see "tensorflow.python.ops.control_flow_ops.GradLoopState#GetRealValue"
   */
  internal fun getRealValue(value: Output): Output {
    TODO()
  }
}