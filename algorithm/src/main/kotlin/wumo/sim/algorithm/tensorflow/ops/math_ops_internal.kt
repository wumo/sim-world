package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.binaryOp

fun TF.igammaGradA(a: Tensor, x: Tensor, name: String = "IgammaGradA") =
    binaryOp("IgammaGradA", a, x, name)

fun TF.invGrad(y: Tensor, dy: Tensor, name: String = "InvGrad") =
    binaryOp("InvGrad", y, dy, name)

fun TF.reciprocalGrad(y: Tensor, dy: Tensor, name: String = "ReciprocalGrad") =
    binaryOp("ReciprocalGrad", y, dy, name)

fun TF.rsqrtGrad(y: Tensor, dy: Tensor, name: String = "RsqrtGrad") =
    binaryOp("RsqrtGrad", y, dy, name)

fun TF.sigmoidGrad(y: Tensor, dy: Tensor, name: String = "SigmoidGrad") =
    binaryOp("SigmoidGrad", y, dy, name)

fun TF.sqrtGrad(y: Tensor, dy: Tensor, name: String = "SqrtGrad") =
    binaryOp("SqrtGrad", y, dy, name)

fun TF.tanhGrad(y: Tensor, dy: Tensor, name: String = "TanhGrad") =
    binaryOp("TanhGrad", y, dy, name)