package wumo.sim.algorithm.util.c_api.core

import org.tensorflow.framework.DataType
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.c_api.Operation
import wumo.sim.algorithm.util.c_api.TF_C

fun TF_C.placeholder(shape: Dimension, dtype: DataType = DataType.DT_FLOAT, name: String = "placeholder"): Operation {
  scope(name) {
    return g.opBuilder("Placeholder", contextPath)
        .setAttr("dtype", dtype)
        .setAttr("shape", shape)
        .build()
  }
}