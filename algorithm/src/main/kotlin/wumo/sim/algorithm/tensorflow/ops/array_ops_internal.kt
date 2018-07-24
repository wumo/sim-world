package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.unaryOp
import wumo.sim.util.a

fun TF.broadcastGradientArgs(s0: Tensor, s1: Tensor, name: String = "BroadcastGradientArgs"): Array<Tensor> {
  val v = g.nodeBuilder("BroadcastGradientArgs", ctxNs.getUniqueFullName(name))
      .addInput(s0)
      .addInput(s1)
      .build()
  return a(Tensor(v, 0), Tensor(v, 1))
}

fun TF.mirrorPadGrad(input: Tensor, paddings: Tensor, mode: String, name: String = "MirrorPadGrad"): Tensor {
  val v = g.nodeBuilder("MirrorPadGrad", ctxNs.getUniqueFullName(name))
      .addInput(input)
      .addInput(paddings)
      .setAttr("mode", mode)
      .build()
  return Tensor(v, 0)
}

fun TF.refIdentity(input: Tensor, name: String = "RefIdentity") =
    unaryOp("RefIdentity", input, name)