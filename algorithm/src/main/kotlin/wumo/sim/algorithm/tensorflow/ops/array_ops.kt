package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.Scope
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor

fun TF.identity(input: Tensor, name: String = "Identity"): Tensor {
  val v = g.nodeBuilder("VariableV2", ctx.getUniqueFullName(name))
      .addInput(input)
      .build()
  return Tensor(v, 0, input.dtype)
}