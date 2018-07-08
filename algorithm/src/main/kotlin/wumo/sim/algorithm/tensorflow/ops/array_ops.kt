package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.binaryOp
import wumo.sim.algorithm.tensorflow.unaryOp
import wumo.sim.algorithm.util.Dimension

fun TF.identity(input: Tensor, name: String = "Identity") =
    unaryOp("Identity", input, name)

fun TF.placeholder(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Placeholder"): Tensor {
  val p = g.nodeBuilder("Placeholder", ctx.getUniqueFullName(name))
      .setAttrType("dtype", dtype)
      .setAttr("shape", shape)
      .build()
  return Tensor(p, 0, dtype)
}

fun TF.zerosLike(x: Tensor, name: String = "ZerosLike") =
    unaryOp("ZerosLike", x, name)

fun TF.onesLike(x: Tensor, name: String = "OnesLike") =
    unaryOp("OnesLike", x, name)

fun TF.zeros(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  subscope(name) {
    val zero = when (dtype) {
      DT_STRING -> ""
      else -> 0
    }
    return if (shape.numElements() < 1000)
      const(shape, dtype, zero, ctx.useContextName())
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, zero), ctx.useContextName())
    }
  }
}

fun TF.ones(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  subscope(name) {
    return if (shape.numElements() < 1000)
      const(shape, dtype, 1, ctx.useContextName())
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, 1), ctx.useContextName())
    }
  }
}

fun TF.fill(dims: Tensor, value: Tensor, name: String = "Fill") =
    binaryOp("Fill", dims, value, name, value.dtype)

fun TF.reshape(tensor: Tensor, shape: Tensor, name: String = "Reshape") =
    binaryOp("Reshape", tensor, shape, name)

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