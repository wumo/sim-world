package wumo.sim.tensorflow.ops.gradients

import wumo.sim.tensorflow.ops.Output

fun <JAC_T> computeGradientError(xs: Collection<Output>, ys: Collection<Output>, max_error: MutableCollection<JAC_T>) {

}

fun <JAC_T> computeGradientError(xs: Output, x_init_value: Output, ys: Output, max_error: MutableCollection<JAC_T>) {

}