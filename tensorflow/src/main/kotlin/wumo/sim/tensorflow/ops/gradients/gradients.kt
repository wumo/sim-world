package wumo.sim.tensorflow.ops.gradients

import register_array_grad
import register_control_flow_grad
import register_cudnn_rnn_grad
import register_data_flow_grad
import register_image_grad
import register_manip_grad
import register_math_grad
import register_nn_grad
import register_random_grad
import register_state_grad
import register_tensor_array_grad
import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.control_flow_ops.ControlFlowContext
import wumo.sim.tensorflow.ops.control_flow_ops.ControlFlowState
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops
import wumo.sim.tensorflow.ops.gradients.gradient_ops.AggregationMethod.AddAggregationMethod
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.tensorflow.util.attrValue
import wumo.sim.util.debug
import wumo.sim.util.emptyMutableSet
import wumo.sim.util.lazyLogger
import wumo.sim.util.t2
import java.util.*

object gradient_ops {
  val logger by lazyLogger()
  
  init {
    register_array_grad()
    register_control_flow_grad()
    register_cudnn_rnn_grad()
    register_data_flow_grad()
    register_image_grad()
    register_manip_grad()
    register_math_grad()
    register_nn_grad()
    register_random_grad()
    register_state_grad()
    register_tensor_array_grad()
  }
  
  interface API {
//    fun gradients(y: Output, xs: Collection<Output>): List<Output> {
//      return addSymbolicGradients(listOf(y), xs.toList())
//    }
//
//    fun gradients(ys: List<Output>, xs: Collection<Output>): List<Output> {
//      return addSymbolicGradients(ys, xs.toList())
//    }
    
    /**
     * Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.
     *
     * `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
     * is a list of `Tensor`, holding the gradients received by the
     * `ys`. The list must be the same length as `ys`.
     *
     * `gradients()` adds ops to the graph to output the derivatives of `ys` with
     * respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
     * each tensor is the `sum(dy/dx)` for y in `ys`.
     
     * `grad_ys` is a list of tensors of the same length as `ys` that holds
     * the initial gradients for each y in `ys`.  When `grad_ys` is None,
    we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
    user can provide their own initial `grad_ys` to compute the
    derivatives using a different initial gradient for each y (e.g., if
    one wanted to weight the gradient differently for each value in
    each y).
    
    `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
    with respect to all `xs`. These tensors will not be backpropagated through,
    as though they had been explicitly disconnected using `stop_gradient`.  Among
    other things, this allows computation of partial derivatives as opposed to
    total derivatives. For example:
    
    ```python
    a = tf.constant(0.)
    b = 2 * a
    g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
    ```
    
    Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
    total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
    influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
    equivalent to:
    
    ```python
    a = tf.stop_gradient(tf.constant(0.))
    b = tf.stop_gradient(2 * a)
    g = tf.gradients(a + b, [a, b])
    ```
    
    `stop_gradients` provides a way of stopping gradient after the graph has
    already been constructed, as compared to `tf.stop_gradient` which is used
    during graph construction.  When the two approaches are combined,
    backpropagation stops at both `tf.stop_gradient` nodes and nodes in
    `stop_gradients`, whichever is encountered first.
    
    All integer tensors are considered constant with respect to all `xs`, as if
    they were included in `stop_gradients`.
    
    Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grad_ys: Optional. A `Tensor` or list of tensors the same size as
    `ys` and holding the gradients computed for each y in `ys`.
    name: Optional name to use for grouping all the gradient ops together.
    defaults to 'gradients'.
    colocate_gradients_with_ops: If True, try colocating gradients with
    the corresponding op.
    gate_gradients: If True, add a tuple around the gradients returned
    for an operations.  This avoids some race conditions.
    aggregation_method: Specifies the method used to combine gradient terms.
    Accepted values are constants defined in the class `AggregationMethod`.
    stop_gradients: Optional. A `Tensor` or list of tensors not to differentiate
    through.
    
    Returns:
    A list of `sum(dy/dx)` for each x in `xs`.
     *
     * @see "tensorflow.python.ops.gradients_impl.gradients"
     */
    fun gradients(ys: List<Output>,
                  xs: List<Output>,
                  grad_ys: List<OutputLike>? = null,
                  gateGradients: Boolean = false,
                  aggregationMethod: AggregationMethod = AddAggregationMethod,
                  colocateGradientsWithOps: Boolean = false,
                  name: String = "gradients"): List<OutputLike?> {
      val to_ops = ys.mapTo(mutableSetOf()) { it.op }
      val from_ops = xs.mapTo(mutableSetOf()) { it.op }
      val grad_ops = grad_ys?.mapTo(mutableSetOf()) { it.op } ?: emptyMutableSet<Op>()
      
      val grads = tf.nameScope(name, to_ops + from_ops + grad_ops) {
        // Get a uid for this call to gradients that can be used to help
        // cluster ops for compilation.
        val gradientUID = tf.currentGraph.uniqueName("uid")
        
        // The approach we take here is as follows: Create a list of all ops in the
        // subgraph between the ys and xs.  Visit these ops in reverse order of ids
        // to ensure that when we visit an op the gradients w.r.t its outputs have
        // been collected.  Then aggregate these gradients if needed, call the op's
        // gradient function, and add the generated gradients to the gradients for
        // its input.
        
        // Initialize the pending count for ops in the connected subgraph from ys
        val toOps = if (ys.size > 1)
          ys.mapTo(mutableSetOf()) { y ->
            if (y.consumers.isNotEmpty()) tf.identity(y).op else y.op
          }
        else
          to_ops
        
        // `pendingCounts(op)` is a count-down counter for the expected gradients to accumulate for `op`. When
        // `pendingCounts(op)` becomes zero, we have collected all the backpropagation gradients for all outputs of `op`.
        val (pendingCount, loopState) = initialPendingCounts(toOps, from_ops, colocateGradientsWithOps)
        
        // Iterate over the collected ops.
        //
        // grads: op => list of gradients received on each output endpoint of the
        // op.  The gradients for each endpoint are initially collected as a list.
        // When it is time to call the op's gradient function, for each endpoint we
        // aggregate the list of received gradients into a Add() Operation if there
        // is more than one.
        val grads = mutableMapOf<Op, MutableList<MutableList<OutputLike?>>>()
        
        // Add the initial gradients for the ys.
        val grad_ys = initialGradients(ys, grad_ys, colocateGradientsWithOps, gradientUID)
        
        ys.asSequence().zip(grad_ys.asSequence()).forEach { (y, grad_y) ->
          setGradient(grads, y, grad_y)
        }
        
        // `readyOps` keeps track of ops that have been completely processed. We initialize it with the destination ops.
        // We filter the destination ops based on whether one output gradient relies on another output's gradient.
        val readyOps = ArrayDeque<Op>(toOps.filter { pendingCount.getOrDefault(it, 0) == 0 })
        
        loopState?.let {
          it.processUnusedLoopExits(pendingCount, readyOps.toSet())
              .filter { isTrainable(it) }
              .forEach { loopExit ->
                setGradient(grads, loopExit, loopState.zerosLikeForExit(loopExit))
                readyOps += loopExit.op
              }
        }
        
        // Stop ops form the frontier of the forward graph before which back-propagation should stop. Ops in this set will
        // not be differentiated. This set is defined as the subset of `sourceOps` containing ops that have no predecessor
        // in `sourceOps`. An op has predecessors in `sourceOps` if and only if `pendingCounts(op) > 0`.
        val stopOps = from_ops.filterTo(mutableSetOf()) {
          it.inputs.all { pendingCount.getOrDefault(it.op, 0) <= 0 }
        }
        
        while (readyOps.isNotEmpty()) {
          // generate gradient subgraph for op.
          val op = readyOps.removeFirst()
          maybeColocateWith(op, colocateGradientsWithOps, gradientUID) {
            loopState?.enterGradientWhileContext(op, before = true)
            val opGradients = aggregationMethod.aggregateGradients(grads, op, gradientUID)
            loopState?.exitGradientWhileContext(op, before = true)
            
            val hasOutputGradients = opGradients.isNotEmpty()
            val gradientFunction = if (hasOutputGradients && op !in stopOps)
              Registry[op.opType]
            else null
            loopState?.enterGradientWhileContext(op, before = false)
            if (hasOutputGradients && gradientFunction != null) {
              // Note that, the gradient aggregation not computing a value for the i'th output, means that the cost does
              // not depend on output i and therefore the gradient with respect to that output is 0.
              opGradients.withIndex().forEach { (outputIndex, gradient) ->
                // Only floating-point outputs get a zero gradient. Gradient functions should ignore the gradient for
                // other outputs.
                val output = op.outputs[outputIndex]
                if (gradient.isEmpty() && isTrainable(output))
                  opGradients[outputIndex] = mutableListOf(loopState?.zerosLike(op, outputIndex)
                                                               ?: ControlFlowContext.zerosLikeOutsideLoop(op, outputIndex))
              }
              
              // Compute the actual op gradients.
              tf.nameScope("${op.name}_grad") {
                // TODO: [CONTEXT] Add support for original op context.
                val outputGradients = opGradients.map { it.firstOrNull() }
                var inputGradients = maybeCompile(name, op) { gradientFunction(op, outputGradients) }
                if (gateGradients && inputGradients.count { it != null } > 1)
                  tf.device(dev = "") {
                    tf.colocateWithForGradient(mutableSetOf(), gradientUID, ignoreExisting = true) {
                      inputGradients = tf.tuple(inputGradients)
                    }
                  }
                val nInp = op.inputs.size
                val nGrd = inputGradients.size
                assert(nInp == nGrd) { "Gradients size ($nGrd) for op '$op' does not match inputs size ($nInp)." }
                logGradients(op, outputGradients, inputGradients)
                // TODO: Report somehow the non-differentiable ops in the graph. This is currently hard to debug.
                op.inputs.asSequence().zip(inputGradients.asSequence())
                    .forEach { (tIn, inGrad) ->
                      if (inGrad == null) return@forEach
                      if (inGrad is Output && tIn.dataType != RESOURCE)
                        inGrad.setShape(tIn.shape)
                      setGradient(grads, tIn, inGrad)
                    }
              }
            }
            loopState?.exitGradientWhileContext(op, before = false)
          }
          
          // Update the pending counts for the inputs of `op` and enqueue ready ops.
          op.inputs.forEach { input ->
            val count = pendingCount.compute(input.op) { _, count ->
              (count ?: 0) - 1
            }!!
            var ready = count == 0
            if (loopState != null && !ready)
              ready = count > 0 && control_flow_ops.isLoopSwitch(input.op)
            if (ready) {
              if (control_flow_ops.isLoopExit(input.op)) {
                // If `input` is an exit without real gradient, defer processing them.
                loopState?.getGradientLoopState(input.op, before = false)?.let { gradState ->
                  gradState.deferredExits += input
                  gradState.pendingExitsCount -= 1
                  if (gradState.pendingExitsCount == 0) {
                    // We now have all the exits and so we process them.
                    var hasRealGradient = false
                    gradState.deferredExits.forEach { exit ->
                      if (grads[exit.op]?.any { it.any { it != null } } == true) {
                        hasRealGradient = true
                        readyOps += exit.op
                      } else
                        gradState.unusedExits += exit
                    }
                    if (hasRealGradient)
                    // For an unused exit, if it has floating-point outputs, we back-propagate a zero gradient.
                    // Otherwise, we just ignore it.
                      gradState.unusedExits.forEach { exit ->
                        if (isTrainable(exit))
                          setGradient(grads, exit, loopState.zerosLikeForExit(exit))
                        readyOps += exit.op
                      }
                    else
                    // All exits are "unused" and so we use `null` as the gradient.
                      gradState.unusedExits.forEach { readyOps += it.op }
                  }
                }
              } else
                readyOps += input.op
            }
          }
        }
        
        loopState?.postProcess()
        grads
      }
      
      // Collect the aggregated gradients for the requested tensors and return them.
      return xs.map { x ->
        val gradients = grads[x.op]?.get(x.valueIndex)
        if (gradients != null && gradients.size > 1)
          throw IllegalArgumentException("The gradients should have been aggregated by now.")
        gradients?.firstOrNull()
      }
    }
    
  }
  
  /** If `colocateGradientsWithOps` is `true`, then all ops created within `block` will be colocated with `op`.
   *
   * @param  op                       Op to maybe colocate with.
   * @param  colocateGradientsWithOps Boolean value indicating whether to colocate the gradient ops with the original
   *                                  ops.
   * @param  gradientUID              Unique identifier within the graph indicating which invocation of gradients is
   *                                  being executed. Used to cluster ops for compilation.
   * @param  block                    Block of code to execute using the specified colocation ops.
   * @return Return value of `block`.
   */
  private fun <R> maybeColocateWith(op: Op,
                                    colocateGradientsWithOps: Boolean,
                                    gradientUID: String,
                                    block: () -> R): R =
      if (colocateGradientsWithOps)
        tf.colocateWithForGradient(mutableSetOf(op), gradientUID, block = block)
      else
        block()
  
  /** If the op was marked as compiled, this function compiles the calculation in `gradientFunction` (using XLA) and
   * returns the result of `gradientFunction`. Otherwise, it simply returns the result of `gradientFunction`.
   *
   * @param  nameScope        Name native to use for the gradient ops.
   * @param  op               Op whose gradients are being computed.
   * @param  gradientFunction Function that computes the gradients for `op`.
   * @return Created gradients op.
   */
  private fun maybeCompile(nameScope: String, op: Op, gradFn: () -> List<OutputLike?>): List<OutputLike?> {
    // TODO: [FUNCTIONAL] Add extra 'func' argument.
    val cleanNameScope = nameScope.removeSuffix("/").replace('/', '_')
    return try {
      val xlaCompile = op.attrBool("_XlaCompile")
      if (!xlaCompile)
        gradFn()
      else {
        val xlaSeparateCompileGradient = op.attrBool("_XlaSeparateCompiledGradients")
        val xlaScope = op.attrString("_XlaScope")
        // If the gradients are supposed to be compiled separately, we give them an '_XlaScope' name that is based on
        // the name_scope of the gradients. Otherwise, they just inherit the existing '_XlaScope' name, which lets them
        // be merged together with the non-gradient computation.
        val xlaGradientsScope = if (xlaSeparateCompileGradient) "${xlaScope}_grad_$cleanNameScope" else xlaScope
        tf.attrScope("_XlaCompile" to attrValue(xlaCompile), "_XlaScope" to attrValue(xlaGradientsScope)) {
          gradFn()
        }
      }
    } catch (e: Exception) {
      gradFn()
    }
  }
  
  /** Computes initial values for the provided gradients, and checks whether their data types are correct.
   *
   * @param  ys                       Sequence containing the variables corresponding to `dys`.
   * @param  dys                      Sequence containing tensor gradients.
   * @param  colocateGradientsWithOps Boolean value indicating whether to colocate the gradient ops with the original
   *                                  ops.
   * @param  gradientUID              Unique identifier within the graph indicating which invocation of gradients is
   *                                  being executed. Used to cluster ops for compilation.
   * @return Sequence containing the default gradient values.
   * @throws InvalidDataTypeException If the gradient tensor data types are not compatible with the input data types.
   *
   * @see "tensorflow.python.ops.gradients_impl._DefaultGradYs"
   */
  private fun initialGradients(ys: List<Output>,
                               grad_ys: List<OutputLike?>?,
                               colocateGradientsWithOps: Boolean,
                               gradientUID: String): List<OutputLike> {
    val grad_ys = grad_ys ?: List(ys.size) { null }
    if (grad_ys.size != ys.size)
      throw IllegalArgumentException("Passed ${grad_ys.size} grad_ys for ${ys.size} ys")
    return ys.asSequence().zip(grad_ys.asSequence()).withIndex().map { (index, value) ->
      val (y, grad_y) = value
      if (grad_y == null) {
        if (y.dataType.isComplex)
          throw InvalidDataTypeException(
              "Gradients of complex tensors must set 'gradients' (variable.dataType = '${y.dataType}').")
        tf.fill(tf.shape(y),
                tf.const(y.dataType, 1, name = "grad_ys_$index"))
      } else {
        when {
          y.dataType.isFloatingPoint || y.dataType.isInteger ->
            if (!grad_y.dataType.isFloatingPoint && !grad_y.dataType.isInteger)
              throw InvalidDataTypeException(
                  "Gradient data type '${grad_y.dataType}' generated for real or integer-valued tensor '$y' with data type " +
                      "'${y.dataType}' must be real or integer.")
          y.dataType.isComplex ->
            if (!grad_y.dataType.isComplex)
              throw InvalidDataTypeException(
                  "Gradient data type '${grad_y.dataType}' generated for complex-valued tensor '$y' with data type " +
                      "'${y.dataType}' must be complex.")
          else -> {
            throw InvalidDataTypeException(
                "Tensor '$y' with data type '${y.dataType}' must be numeric in order to obtain a default gradient.")
          }
        }
        // Create a gradients tensor in the name native of the gradients. This is required in order for tensor arrays
        // to identify which gradient call a gradient value is coming from.
        when (grad_y) {
          is Output -> tf.identity(grad_y, name = "grad_ys_$index")
          is IndexedSlices ->
            IndexedSlices(tf.identity(grad_y.indices, name = "grad_ys_${index}_indices"),
                          tf.identity(grad_y.values, name = "grad_ys_${index}_values"),
                          grad_y.denseShape.let {
                            if (it == null) it
                            else tf.identity(it, name = "grad_ys${index}_shape")
                          })
          is SparseOutput ->
            SparseOutput(tf.identity(grad_y.indices, name = "grad_ys_${index}_indices"),
                         tf.identity(grad_y.values, name = "grad_ys_${index}_values"),
                         grad_y.denseShape.let {
                           if (it == null) it
                           else tf.identity(it, name = "grad_ys${index}_shape")
                         })
        }
      }
    }.toList()
  }
  
  /** Adds the provided `gradient` to the sequence of `output`'s gradients that have been collected so far.
   *
   * @param  gradients Map where the collected gradients are stored.
   * @param  output    Op output whose gradient is provided.
   * @param  gradient  Gradient of `output` to add to the collected gradients.
   */
  private fun setGradient(
      gradients: MutableMap<Op, MutableList<MutableList<OutputLike?>>>,
      output: Output,
      gradient: OutputLike
  ) {
    val opGradients = gradients.getOrPut(output.op) { output.op.outputs.mapTo(mutableListOf()) { mutableListOf<OutputLike?>() } }
    if (control_flow_ops.isLoopSwitch(output.op))
      opGradients[output.valueIndex] = mutableListOf(gradient)
    else
      opGradients[output.valueIndex].add(gradient)
  }
  
  /** Logs the input and output gradients of the provided op.
   *
   * @param  op              Op.
   * @param  outputGradients Output gradients of op.
   * @param  inputGradients  Input gradients of op.
   */
  private fun logGradients(op: Op, outputGradients: List<OutputLike?>, inputGradients: List<OutputLike?>) {
    logger.debug { "Gradients for op '${op.name}':" }
    logger.debug { "  in  --> ${outputGradients.filter { it != null }.joinToString(", ") { it!!.name }}" }
    logger.debug { "  out --> ${inputGradients.filter { it != null }.joinToString(", ") { it!!.name }}" }
  }
  
  /** Initializes the back-propagation input counts for ops between two sets of ops.
   *
   * 'outputMap(op)' indicates the number of back-propagation inputs to this op.
   *
   * @param  toOps                Set of source ops.
   * @param  fromOps           Set of destination ops.
   * @param  colocateGradientsWithOps Boolean value indicating whether to colocate the gradient ops with the original
   *                                  ops.
   * @return Tuple containing: (1) Map from op to the number of back-propagation inputs to this op, and (2) a control
   *         flow gradient state object which is not `None` if the ops between `sources` and `destinations` contain
   *         control flow loops.
   */
  private fun initialPendingCounts(
      toOps: Set<Op>,
      fromOps: Set<Op>,
      colocateGradientsWithOps: Boolean): t2<MutableMap<Op, Int>, ControlFlowState?> {
    //Mark reachable ops from from_ops.
    val reachedOps = mutableSetOf<Op>()
    markReachedOps(fromOps, reachedOps)
    // X in reached_ops iff X is reachable from from_ops by a path of zero or more
    // backpropagatable tensors.
    
    // Mark ops between `sources' and 'destinations'
    val betweenOps = mutableSetOf<Op>()
    val betweenOpList = mutableListOf<Op>()
    val betweenQueue = ArrayDeque<Op>(toOps)
    while (betweenQueue.isNotEmpty()) {
      val op = betweenQueue.removeFirst()
      if (op in reachedOps) {
        betweenOps += op
        betweenOpList += op
        reachedOps -= op //Clear the boolean so we won't add the inputs again.
        op.inputs.forEach { betweenQueue.addLast(it.op) }
      }
    }
    // X in between_ops iff X is on a path of zero or more backpropagatable tensors
    // between from_ops and to_ops
    val loopState = ControlFlowState.maybeCreate(betweenOps, betweenOpList, colocateGradientsWithOps)
    
    // Initialize pending count for between ops.
    val pendingCount = mutableMapOf<Op, Int>()
    betweenOpList.asSequence()
        .flatMap { it.inputs.asSequence() }
        .map { it.op }
        .filter { it in betweenOps }
        .forEach {
          pendingCount.compute(it) { _, count ->
            (count ?: 0) + 1
          }
        }
    return t2(pendingCount, loopState)
  }
  
  /**
   * Mark all ops reached from "from_ops".
   */
  private fun markReachedOps(fromOps: Set<Op>, reachedOps: MutableSet<Op>) {
    val reachedQueue = ArrayDeque<Op>(fromOps)
    while (reachedQueue.isNotEmpty()) {
      val op = reachedQueue.removeFirst()
      if (op !in reachedOps) {
        reachedOps += op
        op.outputs.forEach { output ->
          if (isBackpropagatable(output))
            reachedQueue.addAll(output.consumers)
        }
      }
    }
  }
  
  private val trainableTypes = setOf(FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, RESOURCE)
  private fun isTrainable(tensor: Output): Boolean =
      tensor.dataType.baseDataType in trainableTypes
  
  private val backpropagatableTypes = setOf(BFLOAT16, RESOURCE, VARIANT)
  private fun isBackpropagatable(tensor: Output): Boolean =
      isTrainable(tensor) || tensor.dataType.baseDataType in backpropagatableTypes
  
  sealed class GatingMethod {
    object NoGating : GatingMethod()
    object OpGating : GatingMethod()
    object GraphGating : GatingMethod()
  }
  
  /**
   * Aggregation method used to combine gradients.
   *
   * Computing partial derivatives can require aggregating gradient contributions. All such aggregation methods are
   * represented as objects extending this trait.
   *
   * @see "tensorflow.python.ops.gradients_impl.AggregationMethod"
   */
  sealed class AggregationMethod {
    
    /** Aggregate the gradients for op `op`.
     *
     * @param  gradients   Map where the collected gradients are stored. The gradient sequences corresponding to `op`
     *                     will be replaced with sequences containing a single element corresponding to the aggregated
     *                     gradient.
     * @param  op          Op whose gradients to aggregate.
     * @param  gradientUID Unique identifier within the graph indicating which invocation of gradients is being
     *                     executed. Used to cluster ops for compilation.
     * @see "tensorflow.python.ops.gradients_impl._AggregatedGrads"
     */
    internal fun aggregateGradients(gradients: MutableMap<Op, MutableList<MutableList<OutputLike?>>>,
                                    op: Op,
                                    gradientUID: String): MutableList<MutableList<OutputLike?>> {
      val opGradients = gradients.getOrDefault(op, mutableListOf())
      return if (control_flow_ops.isLoopSwitch(op))
        opGradients
      else {
        opGradients.withIndex().forEach { (index, grads) ->
          if (grads.size < 2)
            grads
          else
            opGradients[index] = mutableListOf(aggreate(grads.filterNotNullTo(mutableListOf()), gradientUID))
        }
        opGradients
      }
    }
    
    /** Aggregates `values` into a single tensor.
     *
     * @param  values      Sequence of values to aggregate.
     * @param  gradientUID Unique identifier within the graph indicating which invocation of gradients is being
     *                     executed (if any). Used to cluster ops for compilation.
     * @return Aggregated tensor.
     */
    internal abstract fun aggreate(gradients: MutableList<OutputLike>,
                                   gradientUID: String? = null): OutputLike
    
    /** Gradient aggregation method that simply adds up the collected gradients. */
    object AddAggregationMethod : AggregationMethod() {
      
      override fun aggreate(gradients: MutableList<OutputLike>, gradientUID: String?): OutputLike {
        return when {
          gradients.all { it is Output } -> {
            // This function adds op outputs from potentially different devices.
            // We add the tensors of each device separately first, and we then add up the partial results.
            val deviceContributions = gradients
                .groupBy { it.device }
                .toList()
                .sortedBy { it.first }
                .map { (_, outputs) ->
                  tf.colocateWithForGradient(mutableSetOf(gradients[0].op), gradientUID, ignoreExisting = true) {
                    tf.addN(outputs.map { it as Output })
                  }
                }
            tf.addN(deviceContributions)
          }
          gradients.all { it is IndexedSlices } -> {
            TODO()
          }
          else -> throw IllegalArgumentException(
              "The gradients being aggregated need to be all of type 'Output' or 'IndexedSlices'.")
        }
      }
    }
    
    /** Gradient aggregation method that simply adds up the collected gradients, without first waiting for all of them to
     * become available at once.
     *
     * The benefit of using this method is that its inputs can be combined in any order and this can allow the expression
     * to be evaluated with a smaller memory footprint. With this method, it is possible to compute a sum of terms which
     * are much larger than total GPU memory.
     */
    object AccumulateAggregationMethod : AggregationMethod() {
      
      override fun aggreate(gradients: MutableList<OutputLike>, gradientUID: String?): OutputLike {
        return when {
          gradients.all { it is Output } -> tf.accumulateN(gradients.map { it as Output })
          gradients.all { it is IndexedSlices } -> {
            TODO()
          }
          else -> throw IllegalArgumentException(
              "The gradients being aggregated need to be all of type 'Output' or 'IndexedSlices'.")
        }
      }
      
    }
    
  }
  
  /** Registry that contains the gradient functions to be used when creating gradient ops. Gradient functions for all
   * types of ops that are being differentiated need to be registered using either the [[Registry.register]] or the
   * [[Registry.registerNonDifferentiable]] functions. In an attempt to obtain the gradient of an op whose type has no
   * gradient function registered, a [[NoSuchElementException]] will be thrown. */
  object Registry {
    
    private val registry = mutableMapOf<String, GradientFunction?>()
    
    /** Registers the provided gradient function to the gradient function registry.
     *
     * Note that if a gradient function for an op of the same type already exists in the registry, then it will be
     * overriden by the provided gradient function.
     *
     * @param  opType   Op type for which a gradient function is being registered.
     * @param  function Gradient function (takes op and output gradients as inputs and returns the input gradients).
     * @see "tensorflow.python.framework.ops.RegisterGradient"
     */
    fun register(vararg opTypes: String, function: GradientFunction) {
      for (opType in opTypes)
        registry[opType] = function
    }
    
    /** Registers the provided op type as non-differentiable (i.e., having `null` as its registered gradient function).
     *
     * This function should *not* be used for ops that have a well-defined gradient that is not yet implemented. It
     * should only be used when defining a new op type. It may be used for ops such as `size` that are not
     * differentiable.
     *
     * The gradient computed for 'opType' will then propagate zeros.
     *
     * For ops that have a well-defined gradient but are not yet implemented, no declaration should be made, and an error
     * *must* be thrown if an attempt to request their gradient is made.
     *
     * @param  opType Op type to register as non-differentiable.
     */
    fun registerNonDifferentiable(vararg opTypes: String) {
      for (opType in opTypes)
        registry[opType] = null
    }
    
    /** Gets the registered gradient function for the provided op type.
     *
     * @param  opType Op type whose gradient function is being looked up.
     * @return Gradient function registered for the provided op type.
     * @throws NoSuchElementException If no gradient has been registered for the provided op type.
     */
    operator fun get(opType: String): GradientFunction? {
      if (opType !in registry)
        throw NoSuchElementException("No gradient registered for op type '$opType'.")
      return registry[opType]
    }
  }
}
typealias GradientFunction = (Op, List<OutputLike?>) -> List<OutputLike?>

//fun TF.gradients(y: Output, xs: Collection<Output>): List<Output> {
//  val _xs = TF_Output(xs.size.toLong())
//  for ((i, x) in xs.withIndex())
//    _xs.position(i.toLong()).oper(x.op!!.c_op).index(x.valueIndex)
//  val dy = TF_Output(xs.size.toLong())
//  val status = newStatus()
//
//  TF_AddGradients(g.c_graph,
//                  y.asTF_Output(), 1,
//                  _xs.position(0L), xs.size,
//                  null, status,
//                  dy)
//  throwExceptionIfNotOk(status)
//  return MutableList(xs.size) {
//    val output = dy.position(it.toLong())
//    val out_type = TF_OperationOutputType(output)
//    Output(Op(g, output.oper()), output.index())
//  }
//}

//fun TF.gradients(ys: List<Output>, xs: Collection<Output>): List<Output> {
//  val _ys = TF_Output(ys.size.toLong())
//  for ((i, y) in ys.withIndex())
//    _ys.position(i.toLong()).oper(y.op!!.c_op).index(y.valueIndex)
//  val _xs = TF_Output(xs.size.toLong())
//  for ((i, x) in xs.withIndex())
//    _xs.position(i.toLong()).oper(x.op!!.c_op).index(x.valueIndex)
//  val dy = TF_Output(xs.size.toLong())
//  val status = newStatus()
//  TF_AddGradients(g.c_graph,
//                  _ys, 1,
//                  _xs.position(0L), xs.size,
//                  null, status,
//                  dy)
//  throwExceptionIfNotOk(status)
//  return MutableList(xs.size) {
//    val output = dy.position(it.toLong())
//    Output(Op(g, output.oper()), output.index())
//  }
//}

//fun TF.gradientDescentOptimizer(learningRate: Float,
//                                loss: Output,
//                                name: String = "GradientDescent"): Op {
//  nameScope(name) {
//    val dy = gradients(loss, trainables)
//    val alpha = const(learningRate, "learning_rate")
//    val applyGradient = mutableListOf<Op>()
//    for ((i, trainable) in trainables.withIndex())
//      applyGradient += applyGradientDescent(trainable, alpha, dy[i])
//    return noOpDep(applyGradient, scopeNameForOnce())
//  }
//}