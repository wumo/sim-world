package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor


fun TF.applyGradientDescent(v: Tensor, alpha: Tensor, delta: Tensor,
                            name: String = "ApplyGradientDescent") =
    g.nodeBuilder("ApplyGradientDescent", ctx.getUniqueFullName(name))
        .addInput(v)
        .addInput(alpha)
        .addInput(delta)
        .setAttr("use_locking", false)
        .build()