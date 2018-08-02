package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.*

fun TF.broadcastGradientArgs(s0: Tensor, s1: Tensor, name: String = "BroadcastGradientArgs") =
    naryOps("BroadcastGradientArgs", s0.value(), s1.value(), name = name)

fun TF.mirrorPadGrad(input: Tensor, paddings: Tensor, mode: String, name: String = "MirrorPadGrad") =
    naryOp("MirrorPadGrad", input.value(), paddings.value(), name = name) {
      attr("mode", mode)
    }

fun TF.refIdentity(input: Tensor, name: String = "RefIdentity") =
    unaryOp("RefIdentity", input.asRef(), name)