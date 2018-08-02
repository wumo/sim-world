package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.naryOp

fun TF.applyGradientDescent(v: Tensor, alpha: Tensor, delta: Tensor, use_locking: Boolean = false,
                            name: String = "ApplyGradientDescent") =
    naryOp("ApplyGradientDescent", v.asRef(), alpha.value(), delta.value(), name = name) {
      attr("use_locking", use_locking)
    }

fun TF.apply_adam(_v: Tensor, m: Tensor, v: Tensor,
                  beta1_power: Tensor, beta2_power: Tensor,
                  lr: Tensor, beta1: Tensor, beta2: Tensor, epsilon: Tensor,
                  grad: Tensor, use_locking: Boolean = false, use_nesterov: Boolean = false,
                  name: String = "ApplyAdam") =
    naryOp("ApplyAdam", _v.asRef(), m.asRef(), v.asRef(),
           beta1_power.value(), beta2_power.value(), lr.value(), beta1.value(), beta2.value(),
           epsilon.value(), grad.value(), name = name) {
      attr("use_locking", use_locking)
      attr("use_nesterov", use_nesterov)
    }