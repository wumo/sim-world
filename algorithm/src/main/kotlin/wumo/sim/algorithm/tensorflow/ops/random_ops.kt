package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.util.Dimension

fun TF.random_uniform(shape: Dimension,
                      dtype: Int = DT_FLOAT,
                      name: String = "RandomUniform"): Tensor {
  subscope(name) {
    val p = g.nodeBuilder("RandomUniform", ctx.name)
        .setAttrType("dtype", dtype)
        .addInput(const(shape.asLongArray(), "shape"))
        .build()
    return Tensor(p, 0, dtype)
  }
}

fun TF.random_uniform(shape: Dimension,
                      min: Float, max: Float,
                      name: String = "RandomUniform"): Tensor {
  subscope(name) {
    val rand = random_uniform(shape, DT_FLOAT)
    val min = const(min, "min")
    val max = const(max, "max")
    return add(mul(rand, sub(max, min)), min, ctx.useContextName())
  }
}