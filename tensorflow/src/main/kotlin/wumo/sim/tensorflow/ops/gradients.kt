package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.ops.gradient_ops.AggregationMethod.AddAggregationMethod

object gradient_ops {
  interface API {
    fun gradients(y: Output, xs: Collection<Output>): List<Output> {
      return addSymbolicGradients(listOf(y), xs.toList())
    }
    
    fun gradients(ys: List<Output>, xs: Collection<Output>): List<Output> {
      return addSymbolicGradients(ys, xs.toList())
    }
    
    /**
     * @see "tensorflow.python.ops.gradients_impl.gradients"
     */
    fun gradients(ys: List<Output>,
                  xs: List<Output>,
                  grad_ys: List<OutputLike>? = null,
                  gateGradients: Boolean = false,
                  aggregationMethod: AggregationMethod = AddAggregationMethod,
                  colocateGradientsWithOps: Boolean = false,
                  name: String = "gradients"): List<OutputLike?> {
      TODO()
    }
    
  }
  
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