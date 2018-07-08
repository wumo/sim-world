package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor

fun TF.biasAdd(value: Tensor, bias: Tensor, name: String = "BiasAdd"): Tensor {
  val op = g.nodeBuilder("BiadAdd", ctx.getUniqueFullName(name))
      .addInput(value)
      .addInput(bias)
      .setAttr("data_format", "NHWC")
      .build()
  return Tensor(op, 0, value.dtype)
}