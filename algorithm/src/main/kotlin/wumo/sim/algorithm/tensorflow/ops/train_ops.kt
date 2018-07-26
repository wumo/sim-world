package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.naryOp


fun TF.applyGradientDescent(v: Tensor, alpha: Tensor, delta: Tensor, use_locking: Boolean = false,
                            name: String = "ApplyGradientDescent") =
    naryOp("ApplyGradientDescent", v, alpha, delta, name = name) {
      attr("use_locking", use_locking)
    }

fun TF.apply_adam(_v: Tensor, m: Tensor, v: Tensor,
                  beta1_power: Tensor, beta2_power: Tensor,
                  lr: Tensor, beta1: Tensor, beta2: Tensor, epsilon: Tensor,
                  grad: Tensor, use_locking: Boolean = false, use_nesterov: Boolean = false,
                  name: String = "ApplyAdam") =
    naryOp("ApplyAdam", _v, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, name = name) {
      attr("use_locking", use_locking)
      attr("use_nesterov", use_nesterov)
    }