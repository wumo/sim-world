package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.RESOURCE
import wumo.sim.util.NONE
import wumo.sim.util.emptyMutableSet
import wumo.sim.util.t2

/** Control flow context for the while-loop construct.
 *
 * @param  maximumIterations     Optional `INT32` scalar specifying the maximum number of iterations to loop for. If
 *                               `null` (the default), no iteration limit is enforced.
 * @param  parallelIterations    Number of iterations allowed to run in parallel.
 * @param  enableBackPropagation If `true`, back-propagation support is enabled for this while-loop context.
 * @param  swapMemory            If `true`, GPU-CPU memory swapping support is enabled for this while-loop context.
 * @param  gradLoopState    Gradient loop state.
 * @param  name                 Name prefix for this while-loop context.
 *
 */
class WhileContext(
    val maximumIterations: Output? = null,
    val parallelIterations: Int = 10,
    override val backPropagate: Boolean = true,
    val swapMemory: Boolean = false,
    override val gradLoopState: GradientLoopState? = null,
    name: String = "while_context"
) : ControlFlowContext() {
  
  /**`BOOLEAN` tensor used for the loop termination condition. Used in code generation for the gradient computation.*/
  internal var pivot: Output? = null
  /**We use this node to control constants created by the predicate function.*/
  private var pivotForPredicate: Op? = null
  /**We use this node to control constants created by the body lambda.*/
  private var pivotForBody: Op? = null
  /**Enter tensors for loop variables.*/
  internal val loopEnters = mutableSetOf<Output>()
  /**Exit tensors for loop variables.*/
  internal val loopExits = mutableSetOf<Output>()
  
  override val name = tf.currentGraph.uniqueName(name)
  override val controlPivot get() = pivotForBody ?: pivotForPredicate
  override fun whileContext(stopContext: ControlFlowContext?): WhileContext? = this
  
  init {
    require(parallelIterations > 0) { "'parallelIterations' must be a positive integer: $parallelIterations" }
    
  }
  
  override fun addValue(output: Output): Output =
      if (output.name in values)
        externalValues.getOrDefault(output.name, output)
      else {
        values += output.name
        // If we are in a grad context and val is from its forward context,
        // use GetRealValue(), which adds the logic to save the history of
        // val in forward.
        tf.currentControlFlowContext?.whileContext()?.gradLoopState?.let { gradientLoopState ->
          getWhileContext(output.op!!)?.let { forwardContext ->
            if (control_flow_ops.isLoopExit(output.op))
              forwardContext.outerContext?.whileContext()
            else
              forwardContext
          }?.let { forwardContext ->
            if (forwardContext == gradientLoopState.forwardContext) {
              val readValue = gradientLoopState.getRealValue(output)
              externalValues[output.name] = readValue
              return readValue
            }
          }
        }
        val result = outerContext?.addValue(output) ?: output
        //Create an Enter to make `result` known to this loop context.
        val enter = tf.controlDependencies(mutableSetOf()) {
          val enter = tf.enter(result, name, isContant = true, parallelIterations = parallelIterations)
          enter.graph.preventFeeding(enter)
          enter
        }
        // Fix the control inputs and control flow context of these enter ops.
        fixControlInputsAndContext(listOf(enter))
        values += enter.name
        externalValues[output.name] = enter
        enter
      }
  
  override fun addOp(op: Op) {
    // For a reduction op, if op is in a grad context and its input is from
    // its forward context, moving op to the forward context means we would
    // store the tensor after the reduction as opposed to the tensor before
    // reduction, and therefore could significantly reduce memory consumption.
    // For now, we do this only for a few ops.
    if (op.opType in setOf("Shape", "Size", "Rank")) {
      val gradientContext = tf.currentControlFlowContext
      gradientContext?.whileContext()?.gradLoopState?.let { gradLoopState ->
        getWhileContext(op.inputs[0].op!!)?.let { opInputForwardContext ->
          if (opInputForwardContext == gradLoopState.forwardContext) {
            val opInputContext = op.inputs[0].op!!.controlFlowContext
            op.controlFlowContext = opInputContext
            opInputContext?.addInternal(op)
            return
          }
        }
      }
    }
    addInternal(op)
  }
  
  /**
   * We move any external control dependencies of the op to the loop pivot, to
   * ensure they get executed.
   * @see "tensorflow.python.ops.control_flow_ops.WhileContext#_AddOpInternal"
   */
  override fun addInternal(op: Op) {
    val externalInputs = if (op.numInputs == 0) {
      // Remove any external control dependencies on this op.
      val (controlInputs, externalInputs) = removeExternalControlEdges(op)
      // Add a control edge from the control pivot to this op.
      if (controlInputs.isEmpty())
        controlPivot?.let { op.addControlInput(it) }
      op.outputs.forEach { values += it.name }
      externalInputs
    } else {
      op.inputs.withIndex().forEach { (index, input) ->
        val realInput = addValue(input)
        if (realInput != input)
          op.updateInput(index, realInput)
      }
      // Remove any external control dependencies on this op.
      val (_, externalInputs) = removeExternalControlEdges(op)
      // Add a control dependency to the op if it only depends on loop invariants. That is to prevent loop invariants
      // from enabling ops that should not be executed.
      maybeAddControlDependency(op)
      op.outputs.forEach { values += it.name }
      externalInputs
    }
    // TODO: [CONTROL_FLOW] Stop ignoring ops with no outputs.
    // Use an identity to pull control inputs as data inputs. Note that we ignore ops which do not have any outputs.
    tf.controlDependencies(emptyMutableSet()) {
      enter()
      val externalInputs = externalInputs.map { op ->
        tf.identity(op.outputs[0]).op!!
      }
      exit()
      externalInputs
    }.let { op.addControlInputs(it) }
    
    if (outerContext != null || !control_flow_ops.isLoopExit(op)) {
      op.graph.preventFetching(op)
      op.outputs.forEach { op.graph.preventFeeding(it) }
    }
  }
  
  /**
   * Add a control input to the op if it only depends on loop invariants.
   */
  private fun maybeAddControlDependency(op: Op) {
    //Determines if `op` needs a control dependency.
    if (op.controlInputs.isEmpty() &&
        ((op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient")
            || op.inputs.all { control_flow_ops.isLoopConstantEnter(it.op!!) }))
      controlPivot?.let { op.addControlInput(it) }
  }
  
  private fun fixControlInputsAndContext(values: List<OutputLike>) {
    values.forEach {
      val outputs = when (it) {
        is Output -> setOf(it)
        is IndexedSlices ->
          if (it.denseShape != null)
            setOf(it.indices, it.values, it.denseShape)
          else
            setOf(it.indices, it.values)
        is SparseOutput ->
          if (it.denseShape != null)
            setOf(it.indices, it.values, it.denseShape)
          else
            setOf(it.indices, it.values)
      }
      outputs.forEach {
        val input = it.op!!.inputs[0]
        val outerControlInputs = ops.controlDependencies(setOf(input)).filter { isInOuterContext(it) }
        it.op.controlFlowContext = this
        it.op.addControlInputs(outerControlInputs)
      }
    }
  }
  
  /** Adds a loop that counts the number of iterations.
   *
   * This is added to the forward loop at the time when we start to create the loop for the back-propagation gradient
   * computation. It is called in the outer context of this forward context.
   *
   * The pseudocode is: `n = 0; while (pivot) { n++; }`
   *
   * Note that a control dependency is added to `n` to ensure the correct execution order of stack push ops.
   *
   * @param  outerGradientLoopState Outer gradient loop state (`None` if not nested).
   * @return Tuple containing the number of iterations taken by the forward loop and the loop index.
   */
  internal fun addForwardLoopCounter(
      outerGradientLoopState: GradientLoopState?): t2<Output, Output> {
    val n = tf.const(0, name = "f_count")
    outerGradientLoopState?.let {
      // Force the stack pushes of i-th execution of an inner loop to be ordered
      // before the pushes of (i+1)-th execution of the same inner loop.
      n.op.addControlInput(it.forwardIndex.op.inputs[0].op)
    }
    enter()
    values += n.name
    val enterN = tf.enter(n, name, false, parallelIterations, name = "f_count")
    loopEnters += enterN
    
    val mergeN = control_flow_ops.merge(listOf(enterN, enterN))[0]
    val switchN = control_flow_ops.switch(mergeN, pivot!!)
    
    val index = switchN[1] + 1
    val nextN = tf.nextIteration(index)
    mergeN.op.updateInput(1, nextN)
    
    val exitN = tf.exit(switchN[0], "f_count").toOutput()
    loopExits += exitN
    exitResult(listOf(exitN))
    exit()
    return t2(exitN, nextN)
  }
  
  /** Adds the back-propagation loop that counts the number of iterations.
   *
   * This is added to the back-propagation loop. It is used to control the loop termination of the back-propagation
   * loop. It is called in the outer context of this gradient context.
   *
   * The pseudocode is: `n = count; while (n >= 1) { n--; }`
   *
   * Note that a control dependency is added to the final exit op to ensure the correct execution order of stack pop
   * ops.
   *
   * @param  count              Number of iterations for the back-propagation loop.
   * @param  outerGradientLoopState Outer gradient loop state (`None` if not nested).
   * @return Loop index.
   * @see "tensorflow.python.ops.control_flow_ops.WhileContext#AddBackpropLoopCounter"
   */
  internal fun addBackwardLoopCounter(
      count: Output, outerGradientLoopState: GradientLoopState?): Output {
    val one = tf.const(1, name = "b_count")
    enter()
    values += count.name
    val enterC = tf.enter(count, name, false, parallelIterations, name = "b_count")
    loopEnters += enterC
    val mergeCount = control_flow_ops.merge(listOf(enterC, enterC))[0]
    
    pivotForPredicate = mergeCount.op
    pivot = tf.loopCond(tf.greaterEqual(mergeCount, one), name = "b_count")
    val switchC = control_flow_ops.switch(mergeCount, pivot!!)
    
    val indexC = switchC[1] - one
    pivotForBody = indexC.op
    val nextCount = tf.nextIteration(indexC)
    mergeCount.op.updateInput(1, nextCount)
    
    val exitC = tf.exit(switchC[0], name = "b_count")
    loopExits += exitC
    
    // Force the stack pops of the i-th execution of an inner loop to be ordered before the pops of the (i+1)-th
    // execution of the same inner loop.
    outerGradientLoopState?.backwardSync?.addControlInput(exitC.op)
    
    exitResult(listOf(exitC))
    exit()
    return nextCount
  }
  
  /** Adds an accumulation loop for every loop invariant.
   *
   * This is added to the back-propagation loop. It is used to accumulate partial gradients within each loop iteration.
   * It is called when in the gradient while context.
   *
   * The pseudocode is: `acc = 0.0; while (pivot) { acc += grad; }`
   *
   * @param  op       Enter op for a loop invariant.
   * @param  gradient Partial gradient of an iteration for a loop invariant.
   * @return Gradient for a loop invariant.
   */
  internal fun <T : OutputLike> addBackwardAccumulator(op: Op, gradient: T): T =
      when (gradient) {
        is Output -> {
          exit()
          // We create a zeros tensor with the right shape for the accumulator. If we don't know the full shape
          // statically, we will have to get the shape dynamically from the forward inference. Getting the shape right for
          // the zeros is only needed for the base case when the loop exits without running any iterations.
          val shape = gradient.shape
          val acc: Output =
              if (shape.isFullyDefined) {
                outerContext?.enter()
                val acc = tf.zerosLike(gradient, name = "b_acc")
                outerContext?.exit()
                acc
              } else {
                val value = op.inputs[0]
                // TODO: !!! [CONTROL_FLOW] Is this even necessary for obtaining the shape?
                when {
                  outerContext is WhileContext && outerContext.gradLoopState != null -> {
                    // We are in a nested while loop.
                    val forwardContext = outerContext.gradLoopState!!.forwardContext
                    forwardContext.outerContext?.enter()
                    val zerosShape = tf.shape(value)
                    forwardContext.outerContext?.exit()
                    val outerGradientLoopState = outerContext.gradLoopState!!.outerGradState!!
                    val historyZerosShape = outerGradientLoopState.addForwardAccumulator(zerosShape)
                    outerContext.enter()
                    val realShape = outerGradientLoopState.addBackwardAccumulatedValue(historyZerosShape, zerosShape)
                    val acc = tf.zeros(realShape, gradient.dataType)
                    outerContext.exit()
//                  acc.setShape(gradient.shape)
                    acc
                  }
                  else -> {
                    outerContext?.enter()
                    val zerosShape = tf.shape(value)
                    val acc = tf.zeros(zerosShape, gradient.dataType)
                    outerContext?.exit()
                    // TODO: [CONTROL_FLOW] Figure out if this is necessary.
                    // acc.setShape(g.shape)
                    acc
                  }
                }
              }
          enter()
          values += acc.name
          val enterAcc = tf.enter(acc, name, false, parallelIterations, name = "b_acc")
          loopEnters += enterAcc
          val mergeAcc = control_flow_ops.merge(listOf(enterAcc, enterAcc))[0]
          val switchAcc = control_flow_ops.switch(mergeAcc, pivot!!)
          
          val addAcc = switchAcc[1] + gradient
          val nextAcc = tf.nextIteration(addAcc)
          mergeAcc.op.updateInput(1, nextAcc)
          
          val exitAcc = tf.exit(switchAcc[0], "b_acc")
          loopExits += exitAcc
          exitResult(listOf(exitAcc))
          exitAcc
        }
        is IndexedSlices -> {
          TODO()
        }
        else -> NONE()
      } as T
  
  /** Returns the shape of `value` of the shape of the variable it points to. */
  private fun resourceSafeShape(value: Output): Output =
      if (value.dataType == RESOURCE) {
        var v = value
        while (v.op.inputs.isNotEmpty())
          v = v.op.inputs[0]
        v.op.attrShape("shape")
      } else
        tf.shape(value, optimize = false)
  
  private fun isInOuterContext(op: Op): Boolean {
    val opContext = control_flow_ops.getOutputContext(op)
    var outerContext = this.outerContext
    while (outerContext != opContext) {
      if (outerContext == null) return false
      outerContext = outerContext.outerContext
    }
    return true
  }
  
  companion object {
    fun getWhileContext(op: Op) = op.controlFlowContext?.whileContext()
  }
}