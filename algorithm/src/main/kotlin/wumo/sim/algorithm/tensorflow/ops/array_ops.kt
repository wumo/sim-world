package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.util.Dimension

fun TF.identity(input: Tensor, name: String = "Identity"): Tensor {
  val v = g.nodeBuilder("VariableV2", ctx.getUniqueFullName(name))
      .addInput(input)
      .build()
  return Tensor(v, 0, input.dtype)
}

fun TF.placeholder(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Placeholder"): Tensor {
  val p = g.nodeBuilder("Placeholder", ctx.getUniqueFullName(name))
      .setAttrType("dtype", dtype)
      .setAttr("shape", shape)
      .build()
  return Tensor(p, 0, dtype)
}
