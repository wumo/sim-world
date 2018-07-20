package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.throwExceptionIfNotOk

fun TF.gradients(y: Tensor, xs: List<Tensor>): List<Tensor> {
  val xs = TF_Output(trainables.size.toLong())
  for ((i, x) in trainables.withIndex())
    xs.position(i.toLong()).oper(x.op.c_op).index(x.value_index)
  val dy = TF_Output(trainables.size.toLong())
  val status = newStatus()
  TF_AddGradients(g.c_graph,
                  y.asTF_Output(), 1,
                  xs.position(0L), trainables.size,
                  null, status,
                  dy)
  throwExceptionIfNotOk(status)
  return MutableList(trainables.size) {
    val output = dy.position(it.toLong())
    val out_type = TF_OperationOutputType(output)
    Tensor(Operation(g, output.oper()), output.index())
  }
}

fun TF.gradients(ys: List<Tensor>, xs: List<Tensor>): List<Tensor> {
  val _ys = TF_Output(ys.size.toLong())
  for ((i, y) in ys.withIndex())
    _ys.position(i.toLong()).oper(y.op.c_op).index(y.value_index)
  val xs = TF_Output(trainables.size.toLong())
  for ((i, x) in trainables.withIndex())
    xs.position(i.toLong()).oper(x.op.c_op).index(x.value_index)
  val dy = TF_Output(trainables.size.toLong())
  val status = newStatus()
  TF_AddGradients(g.c_graph,
                  _ys, 1,
                  xs.position(0L), trainables.size,
                  null, status,
                  dy)
  throwExceptionIfNotOk(status)
  return MutableList(trainables.size) {
    val output = dy.position(it.toLong())
    Tensor(Operation(g, output.oper()), output.index())
  }
}

//fun TF.gradientDescentOptimizer(learningRate: Float,
//                                loss: Tensor,
//                                name: String = "GradientDescent"): Operation {
//  name_scope(name) {
//    val dy = gradients(loss, trainables)
//    val alpha = const(learningRate, "learning_rate")
//    val applyGradient = mutableListOf<Operation>()
//    for ((i, trainable) in trainables.withIndex())
//      applyGradient += applyGradientDescent(trainable, alpha, dy[i])
//    return noOpDep(applyGradient, scopeNameForOnce())
//  }
//}