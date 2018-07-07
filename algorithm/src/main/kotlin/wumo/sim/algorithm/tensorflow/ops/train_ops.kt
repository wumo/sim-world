package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF

fun TF.applyGradientDescent(v: TF_Output, alpha: TF_Output, delta: TF_Output,
                            name: String = "ApplyGradientDescent") =
    g.nodeBuilder("ApplyGradientDescent", ctx.getUniqueFullName(name))
        .addInput(v)
        .addInput(alpha)
        .addInput(delta)
        .setAttr("use_locking", false)
        .build()