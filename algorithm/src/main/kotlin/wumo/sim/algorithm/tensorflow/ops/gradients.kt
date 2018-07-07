package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.throwExceptionIfNotOk

fun TF.gradientDescentOptimizer(learningRate: Float,
                                loss: Tensor,
                                name: String = "GradientDescent"): Operation {
  subscope(name) {
    val xs = TF_Output(trainables.size.toLong())
    for ((i, x) in trainables.withIndex())
      xs.position(i.toLong()).oper(x.op.c_op).index(x.value_index)
    val dy = TF_Output(trainables.size.toLong())
    val status = newStatus()
    TF_AddGradients(g.c_graph,
                    loss.asTF_Output(), 1,
                    xs.position(0L), trainables.size,
                    null, status,
                    dy)
    throwExceptionIfNotOk(status)
    val alpha = const(learningRate, "learning_rate")
    val applyGradient = mutableListOf<Operation>()
    for ((i, trainable) in trainables.withIndex())
      applyGradient += applyGradientDescent(trainable.asTF_Output(), alpha.asTF_Output(), dy.position(i.toLong()))
    return noOpDep(applyGradient, ctx.useContextName())
  }
}