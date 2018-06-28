package wumo.sim.algorithm.util.c_api.random_ops

import org.tensorflow.framework.DataType
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.c_api.Operation
import wumo.sim.algorithm.util.c_api.TF_C
import wumo.sim.algorithm.util.c_api.core.const
import wumo.sim.algorithm.util.c_api.math_ops.add
import wumo.sim.algorithm.util.c_api.math_ops.mul
import wumo.sim.algorithm.util.c_api.math_ops.sub

fun TF_C.random_uniform(shape: Dimension, dtype: DataType = DataType.DT_FLOAT,
                        name: String = "random_uniform"): Operation {
  scope(name) {
    return g.opBuilder("RandomUniform", contextPath)
        .setAttr("dtype", dtype)
        .addInput(const(shape.asLongArray(), "shape"))
        .build()
  }
}

fun TF_C.random_uniform(shape: Dimension, min: Float, max: Float,
                        name: String = "random_uniform"): Operation {
  scope(name) {
    val rand = random_uniform(shape, DataType.DT_FLOAT)
    val min = const(min, name = "min")
    val max = const(max, name = "max")
    return add(mul(rand, sub(max, min)), min, useContextName)
  }
}