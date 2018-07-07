package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
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

fun TF.zerosLike(x: Tensor, name: String = "ZerosLike"): Tensor {
  val v = g.nodeBuilder("ZerosLike", ctx.getUniqueFullName(name))
      .addInput(x)
      .build()
  return Tensor(v, 0, x.dtype)
}

fun TF.onesLike(x: Tensor, name: String = "OnesLike"): Tensor {
  val v = g.nodeBuilder("OnesLike", ctx.getUniqueFullName(name))
      .addInput(x)
      .build()
  return Tensor(v, 0, x.dtype)
}

fun TF.slice(input: Tensor, begin: Tensor, size: Tensor, name: String = "Slice")
    : Tensor {
  val v = g.nodeBuilder("Slice", ctx.getUniqueFullName(name))
      .addInput(input)
      .addInput(begin)
      .addInput(size)
      .build()
  return Tensor(v, 0, input.dtype)
}

fun TF.oneHot(indices: Tensor, depth: Tensor, on_value: Tensor, off_value: Tensor,
              name: String = "OneHot")
    : Tensor {
  val v = g.nodeBuilder("OneHot", ctx.getUniqueFullName(name))
      .addInput(indices)
      .addInput(depth)
      .addInput(on_value)
      .addInput(off_value)
      .build()
  return Tensor(v, 0, on_value.dtype)
}