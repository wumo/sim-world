package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.core.InvalidDataTypeException
import wumo.sim.tensorflow.ops.control_flow_ops.ControlFlowState
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops
import wumo.sim.tensorflow.ops.gradient_ops.AggregationMethod.AddAggregationMethod
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.emptyMutableSet
import wumo.sim.util.t2
import java.util.*

object gradient_ops {
  interface API {
    fun gradients(y: Output, xs: Collection<Output>): List<Output> {
      return addSymbolicGradients(listOf(y), xs.toList())
    }
    
    fun gradients(ys: List<Output>, xs: Collection<Output>): List<Output> {
      return addSymbolicGradients(ys, xs.toList())
    }
    
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
      val to_ops = ys.mapTo(mutableSetOf()) { it.op!! }
      val from_ops = xs.mapTo(mutableSetOf()) { it.op!! }
      val grad_ops = grad_ys?.mapTo(mutableSetOf()) { it.op!! } ?: emptyMutableSet<Op>()
      
      ys.mapTo(mutableSetOf()) { it.op }
      
      tf.nameScope(name, to_ops + from_ops + grad_ops) {
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
        val to_ops = if (ys.size > 1)
          ys.mapTo(mutableSetOf()) { y ->
            if (y.consumers.isNotEmpty()) tf._identity(y).op!! else y.op!!
          }
        else
          to_ops
        
        // `pendingCounts(op)` is a count-down counter for the expected gradients to accumulate for `op`. When
        // `pendingCounts(op)` becomes zero, we have collected all the backpropagation gradients for all outputs of `op`.
        val (pendingCount, loopState) = initialPendingCounts(to_ops, from_ops, colocateGradientsWithOps)
        
        // Iterate over the collected ops.
        //
        // grads: op => list of gradients received on each output endpoint of the
        // op.  The gradients for each endpoint are initially collected as a list.
        // When it is time to call the op's gradient function, for each endpoint we
        // aggregate the list of received gradients into a Add() Operation if there
        // is more than one.
        val grads = mutableMapOf<Op, MutableList<MutableList<OutputLike>>>()
        
        // Add the initial gradients for the ys.
        val grad_ys = initialGradients(ys, grad_ys, colocateGradientsWithOps, gradientUID)
        
        ys.asSequence().zip(grad_ys.asSequence()).forEach { (y, grad_y) ->
          setGradient(grads, y, grad_y)
        }
        
        // `readyOps` keeps track of ops that have been completely processed. We initialize it with the destination ops.
        // We filter the destination ops based on whether one output gradient relies on another output's gradient.
        val readyOps = ArrayDeque<Op>(to_ops.filter { pendingCount.getOrDefault(it, 0) == 0 })
        
        loopState?.let {
        }
      }
      TODO()
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
        tf._fill(tf.shape(y),
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
        // Create a gradients tensor in the name scope of the gradients. This is required in order for tensor arrays
        // to identify which gradient call a gradient value is coming from.
        when (grad_y) {
          is Output -> tf._identity(grad_y, name = "grad_ys_$index")
          is IndexedSlices ->
            IndexedSlices(tf._identity(grad_y.indices, name = "grad_ys_${index}_indices"),
                          tf._identity(grad_y.values, name = "grad_ys_${index}_values"),
                          grad_y.denseShape.let {
                            if (it == null) it
                            else tf._identity(it, name = "grad_ys${index}_shape")
                          })
          is SparseOutput ->
            SparseOutput(tf._identity(grad_y.indices, name = "grad_ys_${index}_indices"),
                         tf._identity(grad_y.values, name = "grad_ys_${index}_values"),
                         grad_y.denseShape.let {
                           if (it == null) it
                           else tf._identity(it, name = "grad_ys${index}_shape")
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
      gradients: MutableMap<Op, MutableList<MutableList<OutputLike>>>,
      output: Output,
      gradient: OutputLike
  ) {
    val opGradients = gradients.getOrPut(output.op!!) { output.op.outputs.mapTo(mutableListOf()) { mutableListOf<OutputLike>() } }
    if (control_flow_ops.isLoopSwitch(output.op))
      opGradients[output.value_index] = mutableListOf(gradient)
    else
      opGradients[output.value_index].add(gradient)
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
        op.inputs.forEach { betweenQueue.addLast(it.op!!) }
      }
    }
    // X in between_ops iff X is on a path of zero or more backpropagatable tensors
    // between from_ops and to_ops
    val loopState = ControlFlowState.maybeCreate(betweenOps, betweenOpList, colocateGradientsWithOps)
    
    // Initialize pending count for between ops.
    val pendingCount = mutableMapOf<Op, Int>()
    betweenOpList.asSequence()
        .flatMap { it.inputs.asSequence() }
        .map { it.op!! }
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
      tensor.dataType.base_dtype in trainableTypes
  
  private val backpropagatableTypes = setOf(BFLOAT16, RESOURCE, VARIANT)
  private fun isBackpropagatable(tensor: Output): Boolean =
      isTrainable(tensor) || tensor.dataType.base_dtype in backpropagatableTypes
  
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
    
    object AddAggregationMethod : AggregationMethod()
    object AccumulateAggregationMethod : AggregationMethod()
    
  }
}

//fun TF.gradients(y: Output, xs: Collection<Output>): List<Output> {
//  val _xs = TF_Output(xs.size.toLong())
//  for ((i, x) in xs.withIndex())
//    _xs.position(i.toLong()).oper(x.op!!.c_op).index(x.value_index)
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
//    _ys.position(i.toLong()).oper(y.op!!.c_op).index(y.value_index)
//  val _xs = TF_Output(xs.size.toLong())
//  for ((i, x) in xs.withIndex())
//    _xs.position(i.toLong()).oper(x.op!!.c_op).index(x.value_index)
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