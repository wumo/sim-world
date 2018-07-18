package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.unaryOp

fun TF.biasAdd(value: Tensor, bias: Tensor, name: String = "BiasAdd"): Tensor {
  val op = g.nodeBuilder("BiasAdd", ctx.getUniqueFullName(name))
      .addInput(value)
      .addInput(bias)
      .setAttr("data_format", "NHWC")
      .build()
  return Tensor(op, 0)
}

fun TF.relu(features: Tensor, name: String = "Relu") =
    unaryOp("Relu", features, name)

fun TF.softmax(logits: Tensor, name: String = "Softmax") =
    unaryOp("Softmax", logits, name)