package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor


fun TF.applyGradientDescent(v: Tensor, alpha: Tensor, delta: Tensor,
                            name: String = "ApplyGradientDescent") =
    g.nodeBuilder("ApplyGradientDescent", ctxNs.getUniqueFullName(name))
        .addInput(v)
        .addInput(alpha)
        .addInput(delta)
        .attr("use_locking", false)
        .build()