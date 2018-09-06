package wumo.sim.tensorflow.ops.control_flow_ops

import wumo.sim.tensorflow.core.ShapeMismatchException
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.basic.minus
import wumo.sim.tensorflow.ops.basic.plus
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.RESOURCE
import wumo.sim.util.*

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
          getWhileContext(output.op)?.let { forwardContext ->
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
        getWhileContext(op.inputs[0].op)?.let { opInputForwardContext ->
          if (opInputForwardContext == gradLoopState.forwardContext) {
            val opInputContext = op.inputs[0].op.controlFlowContext
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
  @Suppress("NAME_SHADOWING")
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
        tf.identity(op.outputs[0]).op
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
            || op.inputs.all { control_flow_ops.isLoopConstantEnter(it.op) }))
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
        val input = it.op.inputs[0]
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
    
    val mergeN = tf.merge(listOf(enterN, enterN))[0]
    val switchN = tf.switch(mergeN, pivot!!)
    
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
    val mergeCount = tf.merge(listOf(enterC, enterC))[0]
    
    pivotForPredicate = mergeCount.op
    pivot = tf.loopCond(tf.greaterEqual(mergeCount, one), name = "b_count")
    val switchC = tf.switch(mergeCount, pivot!!)
    
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
  @Suppress("UNCHECKED_CAST")
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
          val mergeAcc = tf.merge(listOf(enterAcc, enterAcc))[0]
          val switchAcc = tf.switch(mergeAcc, pivot!!)
          
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
    
    /** Returns `true` if `shape2` is a less strict shape than `shape1`, while being compatible with `shape1`. */
    internal fun shapeLessThenOrEqual(shape1: Shape, shape2: Shape): Boolean =
        shape2.rank == -1 ||
            shape1.rank == shape2.rank ||
            shape1.asIntArray()!!.zip(shape2.asIntArray()!!).all {
              it._2 == -1 || it._1 == it._2
            }
    
    /** Sets the shapes of the tensors in `enterTensors` to `shapes` and makes sure that the shape invariants apply.
     *
     * @param  inputTensors Tensors that are inputs to `enterTensors`.
     * @param  enterTensors Tensors whose shapes will be set.
     * @param  shapes       Shapes to use for `enterTensors`.
     * @throws ShapeMismatchException   If any tensor in `inputTensors` has a less specific shape than its corresponding
     *                                  shape in `shapes`.
     * @throws IllegalArgumentException If the types of the input tensors do not match the types of the enter tensors or
     *                                  if the type of either is not supported.
     */
    internal fun setShapeInvariants(inputTensors: List<OutputLike>,
                                    enterTensors: List<OutputLike>,
                                    shapes: List<Shape>) {
      // Check that the shapes of the inputs are less than the shape invariants, and set the shapes of the enter tensors
      // to the shape invariants.
      zip(inputTensors, enterTensors, shapes) { (input, enter, shape) ->
        when {
          input is Output && enter is Output -> {
            if (!shapeLessThenOrEqual(input.shape, shape))
              throw ShapeMismatchException(
                  "The shape invariant specified for '${input.name}' is not compatible with the initial shape of the " +
                      "loop variable. It enters the loop with shape '${input.shape}', but the specified shape invariant " +
                      "is '$shape'.")
            enter.setShape(shape)
          }
          input is IndexedSlices && enter is IndexedSlices -> {
            if (!shapeLessThenOrEqual(input.values.shape, shape))
              throw ShapeMismatchException(
                  "The shape invariant specified for '${input.values.name}' is not compatible the initial shape of the " +
                      "values tensor of these indexed slices. It enters the loop with shape '${input.values.shape}', but " +
                      "the specified shape invariant is '$shape'.")
            enter.values.setShape(shape)
            enter.indices.setShape(Shape(shape[0]))
            if (enter.denseShape != null)
              enter.denseShape.setShape(Shape(shape.rank))
          }
          input is SparseOutput && enter is SparseOutput -> {
            if (!shapeLessThenOrEqual(input.denseShape!!.shape, shape))
              throw ShapeMismatchException(
                  "The shape invariant specified for '${input.denseShape.name}' is not compatible the initial shape of the " +
                      "dense shape tensor of this sparse tensor. It enters the loop with shape '${input.denseShape.shape}', " +
                      " but the specified shape invariant is '$shape'.")
            enter.values.setShape(Shape(-1))
            enter.indices.setShape(Shape(-1, shape.rank))
            enter.denseShape!!.setShape(shape)
          }
          else -> throw IllegalArgumentException(
              "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the input tensor " +
                  "and the enter tensor types must match.")
        }
      }
    }
    
    /** Checks if the shapes of a loop variable satisfy the shape invariants.
     *
     * @param  mergeTensor Tensor representing the initial value of the loop variable.
     * @param  nextTensor  Tensor representing the value of the loop variable after one loop iteration.
     * @throws ShapeMismatchException   If `mergeTensor` has a less specific shape than its corresponding shape in
     *                                  `nextTensor`.
     * @throws IllegalArgumentException If the type of the merge tensor does not match the type of the next tensor or if
     *                                  the type of either is not supported.
     */
    internal fun enforceShapeInvariant(mergeTensor: OutputLike,
                                       nextTensor: OutputLike) {
      when {
        mergeTensor is Output && nextTensor is Output ->
          if (!shapeLessThenOrEqual(nextTensor.shape, mergeTensor.shape))
            throw ShapeMismatchException(
                "The shape for '${mergeTensor.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                    "'${mergeTensor.shape}', but has shape '${nextTensor.shape}' after one iteration. Please provide shape " +
                    "invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' method of " +
                    "the loop variables.")
        mergeTensor is IndexedSlices && nextTensor is IndexedSlices -> {
          val mergeValuesShape = mergeTensor.values.shape
          val mergeIndicesShape = mergeTensor.indices.shape
          val mergeDenseShapeShape = mergeTensor.denseShape?.shape ?: Shape()
          val nextValuesShape = nextTensor.values.shape
          val nextIndicesShape = nextTensor.indices.shape
          val nextDenseShapeShape = nextTensor.denseShape?.shape ?: Shape()
          if (!shapeLessThenOrEqual(nextValuesShape, mergeValuesShape) ||
              !shapeLessThenOrEqual(nextIndicesShape, mergeIndicesShape) ||
              !shapeLessThenOrEqual(nextDenseShapeShape, mergeDenseShapeShape))
            throw ShapeMismatchException(
                "The shape for '${mergeTensor.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                    "'($mergeValuesShape, $mergeIndicesShape, $mergeDenseShapeShape)', but has shape " +
                    "'($nextValuesShape, $nextIndicesShape, $nextDenseShapeShape)' after one iteration. Please provide " +
                    "shape invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' " +
                    "method of the loop variables.")
        }
        mergeTensor is SparseOutput && nextTensor is SparseOutput -> {
          val mergeValuesShape = mergeTensor.values.shape
          val mergeIndicesShape = mergeTensor.indices.shape
          val mergeDenseShapeShape = mergeTensor.denseShape!!.shape
          val nextValuesShape = nextTensor.values.shape
          val nextIndicesShape = nextTensor.indices.shape
          val nextDenseShapeShape = nextTensor.denseShape!!.shape
          if (!shapeLessThenOrEqual(nextValuesShape, mergeValuesShape) ||
              !shapeLessThenOrEqual(nextIndicesShape, mergeIndicesShape) ||
              !shapeLessThenOrEqual(nextDenseShapeShape, mergeDenseShapeShape))
            throw ShapeMismatchException(
                "The shape for '${mergeTensor.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                    "'($mergeValuesShape, $mergeIndicesShape, $mergeDenseShapeShape)', but has shape " +
                    "'($nextValuesShape, $nextIndicesShape, $nextDenseShapeShape)' after one iteration. Please provide " +
                    "shape invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' " +
                    "method of the loop variables.")
        }
        else -> throw IllegalArgumentException(
            "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the merge tensor " +
                "and the next tensor types must match>")
      }
    }
    
    /** Creates a next iteration op for `v` and adds a back edge from `v` to `m`. */
    @Suppress("UNCHECKED_CAST")
    internal fun <T : OutputLike> addNextIterationAndBackEdge(
        m: T, v: T, enforceShapeInvariant: Boolean = true): T =
        when {
          m is Output && v is Output -> {
            val nextV = tf.nextIteration(v)
            if (enforceShapeInvariant)
              enforceShapeInvariant(m, v)
            m.op.updateInput(1, v)
            nextV
          }
          m is IndexedSlices && v is IndexedSlices -> {
            val nextV = tf.nextIteration(v as IndexedSlices)
            m.values.op.updateInput(1, nextV.values)
            m.indices.op.updateInput(1, nextV.indices)
            if (m.denseShape != null) {
              if (nextV.denseShape == null)
                throw  IllegalArgumentException("Output indexed slices '$nextV' must have dense shape information.")
              m.denseShape.op.updateInput(1, nextV.denseShape)
            }
            nextV
          }
          m is SparseOutput && v is SparseOutput -> {
            val nextV = tf.nextIteration(v as SparseOutput)
            m.values.op.updateInput(1, nextV.values)
            m.indices.op.updateInput(1, nextV.indices)
            m.denseShape!!.op.updateInput(1, v.denseShape!!)
            nextV
          }
          else -> throw IllegalArgumentException(
              "Only 'Output', 'IndexedSlices', and 'SparseOutput' are supported. Also, the tensor types must match.")
        } as T
  }
}