package wumo.sim.algorithm.tensorflow.ops.gradients

import wumo.sim.algorithm.tensorflow.Tensor

fun <JAC_T> computeGradientError(xs: Collection<Tensor>, ys: Collection<Tensor>, max_error: MutableCollection<JAC_T>) {

}

fun <JAC_T> computeGradientError(xs: Tensor, x_init_value: Tensor, ys: Tensor, max_error: MutableCollection<JAC_T>) {

}