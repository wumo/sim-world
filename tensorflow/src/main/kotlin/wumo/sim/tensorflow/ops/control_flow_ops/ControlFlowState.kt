package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.IndexedSlices
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.SparseOutput
import wumo.sim.tensorflow.ops.control_flow_ops.ControlFlowContext.Companion.zerosLikeOutsideLoop
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.isLoopSwitch
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.isSwitch
import wumo.sim.tensorflow.tf

/**
 * Maintains the mapping from while-loops to their gradient states.
 *
 * @see "tensorflow.python.ops.control_flow_ops.ControlFlowState"
 */
class ControlFlowState {
  
  private val map: MutableMap<ControlFlowContext, GradientLoopState> = mutableMapOf()
  
  /** Returns the gradient loop state for `op`, if it is in a forward loop context. */
  internal fun getGradientLoopState(op: Op, before: Boolean): GradientLoopState? {
    val forwardContext =
        if (before && control_flow_ops.isLoopExit(op))
          op.controlFlowContext?.outerContext?.whileContext()
        else
          WhileContext.getWhileContext(op)
    return forwardContext?.let {
      map[it]
    }
  }
  
  /** Enters the appropriate while-loop context used for gradient computation. */
  internal fun enterGradientWhileContext(op: Op, before: Boolean) {
    getGradientLoopState(op, before)?.backwardContext?.enter()
  }
  
  /** Exits the appropriate while-loop context used for gradient computation. */
  internal fun exitGradientWhileContext(op: Op, before: Boolean) {
    getGradientLoopState(op, before)?.backwardContext?.exit()
  }
  
  /** Adds the gradient loop state for the while loop that `op` belongs to.
   *
   * Note that `op` must be an exit op, and this method must be called in the same control flow context where
   * `gradients()` is called from.
   *
   * Note that this method modifies `between` and `betweenList`. */
  internal fun addWhileLoopContext(
      op: Op, between: MutableSet<Op>, betweenList: MutableList<Op>) {
    WhileContext.getWhileContext(op)?.let {
      if (it !in map) {
        // This is a new while loop and so we create a new gradient loop state for it.
        val outerForwardContext = it.outerContext?.whileContext()
        val outerGradientLoopState = outerForwardContext?.let { map[it] }
        val gradientLoopState = GradientLoopState(it, outerGradientLoopState)
        map[it] = gradientLoopState
        
        // We need to include all exit ops of a loop for back-propagation.
        gradientLoopState.forwardLoopExits
            .asSequence()
            .filter { it.op !in between }
            .forEach {
              between += it.op
              betweenList += it.op
            }
      }
    }
  }
  
  /** Creates a zeros tensor with the same data type and shape as the provided loop exit.
   *
   * If the result of a loop variable is not used but is involved in computing the result of some needed loop variable,
   * we create a zero-valued tensor that is fed as gradient for the exit node of that loop variable. Note that
   * `value.op` must be an exit op, and this method must be called in the same control flow context where `gradients()`
   * is called from.
   *
   * @param  value Output of an exit op.
   * @return Zeros tensor with the same data type and shape as `value`.
   */
  internal fun zerosLikeForExit(value: Output): Output {
    val forwardContext = value.op.controlFlowContext
    val outerForwardContext = forwardContext?.outerContext?.whileContext()
    val outerGradientLoopState = outerForwardContext?.let { map[it] }
    return if (outerGradientLoopState != null) {
      // This is a nested loop.
      if (value.shape.isFullyDefined) {
        // If the shape is known statically, we just create a zeros tensor with the right shape in the right context.
        outerGradientLoopState.backwardContext.enter()
        val result = tf.zeros(value.shape, value.dataType)
        outerGradientLoopState.backwardContext.exit()
        result
      } else {
        // Only the shape of `value` is needed for back-propagation.
        forwardContext.outerContext.enter()
        val shape = tf.shape(value, optimize = false)
        forwardContext.outerContext.exit()
        // Save the shape to a stack.
        val historyShape = outerGradientLoopState.addForwardAccumulator(shape)
        // Get the shape back from the stack.
        outerGradientLoopState.backwardContext.enter()
        val realShape = outerGradientLoopState.addBackwardAccumulatedValue(historyShape, shape)
        val result = tf.zeros(realShape, value.dataType)
        outerGradientLoopState.backwardContext.exit()
        result
      }
    } else {
      // This is not a nested loop.
      if (value.shape.isFullyDefined)
      // If the shape is known statically, we just create a zeros tensor with the right shape.
        tf.zeros(value.shape, value.dataType)
      else
        tf.zerosLike(value, optimize = false)
    }
  }
  
  /** Creates a zeros tensor with the same data type and shape as the provided op output.
   *
   * If `op` is in a while loop that is part of `gradients()`, this method must be called in its gradient loop context.
   *
   * @param  op    Op.
   * @param  index Op output index.
   * @return Zeros tensor with the same data type and shape as `op.outputs(index)`.
   */
  internal fun zerosLike(op: Op, index: Int): Output? {
    if (isLoopSwitch(op))
      return null
    val deadBranch = isSwitch(op)
    val forwardContext = WhileContext.getWhileContext(op)
    val gradState = forwardContext?.let { map[it] }
        ?: return zerosLikeOutsideLoop(op, index)// `op` is not in a while loop that is part of `gradients()`.
    val opContext = op.controlFlowContext as? CondContext
    val value = op.outputs[index]
    
    return if (value.shape.isFullyDefined) {
      // If the shape is known statically, we just create a zeros tensor with the right shape in the right context.
      var result = tf.zeros(value.shape, value.dataType)
      if (deadBranch) {
        // `op` is a conditional switch and so we guard the zero tensor with a switch.
        val pred = gradState.historyMap[opContext!!.predicate.name]!!
        val branch = opContext.branch
        result = control_flow_ops._switchRefOrTensor(result, pred)[1 - branch]
      }
      result
    } else {
      lateinit var zerosShape: Output
      // Unknown shape and so we keep a history of the shape at runtime.
      if (deadBranch) {
        // `op` is a conditional switch and so we guard the zero tensor with a switch.
        opContext?.let {
          it.outerContext?.enter()
          val value = tf.switch(op.inputs[0], opContext.predicate)[1 - opContext.branch]
          zerosShape = tf.shape(value, optimize = false)
          it.outerContext?.exit()
          value.op.controlFlowContext = opContext
          zerosShape.op.controlFlowContext = opContext
        }
      } else {
        opContext?.enter()
        zerosShape = tf.shape(value, optimize = false)
        opContext?.exit()
      }
      // Add forward accumulator for the shape.
      gradState.backwardContext.exit()
      val historyShape = gradState.addForwardAccumulator(zerosShape, deadBranch)
      gradState.backwardContext.enter()
      
      // Create a zeros tensor with the right shape.
      val realShape = gradState.addBackwardAccumulatedValue(historyShape, zerosShape, deadBranch)
      tf.zeros(realShape, value.dataType)
    }
  }
  
  /** Processes all of the "unused" loop exits.
   *
   * The "unused" exits of the loops are added to `unusedExits`. An exit is unused if its pending count is 0. If there
   * is an exit with a real gradient, all these deferred exits will enter the back-propagation loop with zero gradient.
   * Otherwise, they will enter the back-propagation loop with `None`. As an example, people often write:
   * ```
   *   val loopResult = tf.whileLoop(p, b, Seq(x1, x2))
   *   val result = tf.gradients(loopResult._1, x1)
   * ```
   * The exit node for `x2` is not included because of the between-ness analysis. However, we need to back-propagate
   * `x2` if `x2` is involved in computing `loopResult._1`.
   *
   * @param  pendingCounts  Number of back-propagation inputs for each op.
   * @param  destinationOps Set of destination ops in the gradient computation.
   * @return Set of unused loop exits that we know at this point that we need to back-propagate.
   */
  internal fun processUnusedLoopExits(
      pendingCounts: MutableMap<Op, Int>, destinationOps: Set<Op>): Set<Output> {
    val loopExits = mutableSetOf<Output>()
    map.values.forEach { gradientLoopState ->
      gradientLoopState.forwardLoopExits.asSequence()
          .filter { pendingCounts.getOrDefault(it.op, 0) == 0 }
          .forEach { exit ->
            gradientLoopState.pendingExitsCount -= 1
            if (exit.op !in destinationOps)
              gradientLoopState.unusedExits += exit
            if (gradientLoopState.pendingExitsCount == 0)
              loopExits += gradientLoopState.unusedExits
          }
      // We need to include enter ops in the back-propagation too, for higher-order gradients.
      gradientLoopState.forwardContext.loopEnters.asSequence()
          .map { it.op }
          .filter { pendingCounts.getOrDefault(it, 0) == 0 }
          .forEach {
            pendingCounts[it] = 1
          }
    }
    return loopExits
  }
  
  /** Performs postprocessing at the end of the `gradients()` function call.
   *
   * We have created the gradient graph at this point. So this function can be used to perform any postprocessing on
   * the gradient graph. We currently perform the following postprocessing:
   *
   *   1. Patch the gradient graph if the output of a loop variable doesn't depend on its input.
   *
   * @see "tensorflow.python.ops.control_flow_ops.ControlFlowState#PostProcessing"
   */
  internal fun postProcess() {
    map.values.forEach { gradientLoopState ->
      gradientLoopState.switchMap.values
          .asSequence()
          .flatMap {
            when (it) {
              is Output -> listOf(it)
              is IndexedSlices ->
                if (it.denseShape == null)
                  listOf(it.indices, it.values)
                else
                  listOf(it.indices, it.values, it.denseShape)
              is SparseOutput ->
                if (it.denseShape == null)
                  listOf(it.indices, it.values)
                else
                  listOf(it.indices, it.values, it.denseShape)
            }.asSequence()
          }
          .filter {
            it.op.inputs[0] == it.op.inputs[1]
          }
          .forEach { merge ->
            // The value of this loop variable at iteration i+1 does not depend on its value at iteration i and so we use
            // zeros as the gradients for all iterations > 0.
            val dataType = merge.op.inputs[0].dataType
            val shape = merge.op.inputs[0].shape
            val nextGradientValue = if (shape.isFullyDefined) {
              gradientLoopState.backwardContext.enter()
              // Create a zeros tensor and use it for iterations > 0.
              val gradientValue = tf.zeros(shape, dataType)
              val nextGradientValue = tf.nextIteration(gradientValue)
              gradientLoopState.backwardContext.exit()
              nextGradientValue
            } else {
              // Create a zeros tensor in the outer gradient context.
              val outerGradientContext = gradientLoopState.backwardContext.outerContext
              outerGradientContext?.enter()
              val enterGradient = merge.op.inputs[0].op.inputs[0]
              val gradientShape = tf.shape(enterGradient, optimize = false)
              val gradientValue = tf.zeros(gradientShape, dataType)
              outerGradientContext?.exit()
              // Use the zeros for iterations > 0.
              gradientLoopState.backwardContext.enter()
              val nextGradientValue = tf.nextIteration(gradientValue)
              gradientLoopState.backwardContext.exit()
              nextGradientValue
            }
            merge.op.updateInput(1, nextGradientValue)
          }
    }
  }
  
  companion object {
    /** Creates the state for all the while loops involved in one `gradients()` call.
     *
     * This function creates a [[ControlFlowState]] when there are while loops involved in the `gradients()` call. In
     * `gradients()`, the control flow logic is only invoked when the gradient state is not `None`.
     *
     * Note that this method modifies `between` and `betweenList`.
     *
     * @see "tensorflow.python.ops.control_flow_ops.MaybeCreateControlFlowState"
     */
    internal fun maybeCreate(betweenOps: MutableSet<Op>, betweenOpList: MutableList<Op>,
                             colocateGradientsWithOps: Boolean): ControlFlowState? {
      var loopState: ControlFlowState? = null
      betweenOpList.asSequence()
          .filter { control_flow_ops.isLoopExit(it) }
          .forEach { op ->
            if (loopState == null)
              loopState = ControlFlowState()
            if (colocateGradientsWithOps)
              tf.colocateWith(op) {
                loopState?.addWhileLoopContext(op, betweenOps, betweenOpList)
              }
            else
              loopState?.addWhileLoopContext(op, betweenOps, betweenOpList)
          }
      return loopState
    }
  }
}