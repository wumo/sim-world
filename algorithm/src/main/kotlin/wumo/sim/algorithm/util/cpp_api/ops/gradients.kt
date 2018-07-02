package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.cpp_api.TF_CPP

fun TF_CPP.gradientDescentOptimizer(learningRate: Float,
                                    loss: Output,
                                    name: String = "",
                                    scope: tensorflow.Scope = root): Operation {
  scope.NewSubScope(name).let { s ->
    val node_outputs = OutputVector(loss)
    val node_inputs = OutputVector(*trainables.toTypedArray())
    val node_grad_outputs = OutputVector()
    TF_CHECK_OK(AddSymbolicGradients(s, node_outputs, node_inputs, node_grad_outputs))
    val alpha = const(learningRate, "learning_rate", s)
    val applyGradient = mutableListOf<Output>()
    for ((i, trainable) in trainables.withIndex())
      applyGradient += ApplyGradientDescent(s, Input(trainable), Input(alpha), Input(node_grad_outputs[i.toLong()])).asOutput()
    return noOpDep(applyGradient)
  }
}