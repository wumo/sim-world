package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.binaryOp
import wumo.sim.algorithm.tensorflow.unaryOp
import wumo.sim.algorithm.util.Dimension

fun TF.identity(input: Tensor, name: String = "Identity") =
    unaryOp("Identity", input, name)

fun TF.placeholder(dtype: Int = DT_FLOAT, name: String = "Placeholder"): Tensor {
  val tensor_shape_proto = TensorShapeProto()
  tensor_shape_proto.set_unknown_rank(true)
  val p = g.nodeBuilder("Placeholder", ctx.getUniqueFullName(name))
      .setAttrType("dtype", dtype)
      .setAttr("shape", tensor_shape_proto)
      .build()
  return Tensor(p, 0, dtype)
}

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
      const(shape, dtype, zero, borrowParentName())
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, zero), borrowParentName())
    }
  }
}

fun TF.ones(shape: Dimension, dtype: Int = DT_FLOAT, name: String = "Ones"): Tensor {
  subscope(name) {
    return if (shape.numElements() < 1000)
      const(shape, dtype, 1, borrowParentName())
    else {
      val shape = reshape(const(shape.asLongArray()), const(-1))
      fill(shape, const(dtype, 1), borrowParentName())
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

fun TF.shape(input: Tensor, name: String = "Shape", optimize: Boolean = true): Tensor {
  //TODO SparseTensor
  val out_type = DT_INT32
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.asIntArray(), name)
  val op = g.nodeBuilder("Shape", ctx.getUniqueFullName(name))
      .addInput(input)
      .setAttrType("out_type", out_type)
      .build()
  return Tensor(op, 0, out_type)
}

operator fun Tensor.get(vararg idx: Int): Tensor {
  TODO()
}

class StridedSliceAttrs {
  var begin_mask_: Int = 0
  var end_mask_: Int = 0
  var ellipsis_mask_: Int = 0
  var new_axis_mask_: Int = 0
  var shrink_axis_mask_: Int = 0
}

fun TF.strideSlice(input: Tensor, begin: Tensor, end: Tensor, strides: Tensor,
                   attrs: StridedSliceAttrs = StridedSliceAttrs(),
                   name: String = "StridedSlice"): Tensor {
  val op = g.nodeBuilder("StridedSlice", ctx.getUniqueFullName(name))
      .addInput(input)
      .addInput(begin)
      .addInput(end)
      .addInput(strides)
      .setAttr("begin_mask", attrs.begin_mask_)
      .setAttr("end_mask", attrs.end_mask_)
      .setAttr("ellipsis_mask", attrs.ellipsis_mask_)
      .setAttr("new_axis_mask", attrs.new_axis_mask_)
      .setAttr("shrink_axis_mask", attrs.shrink_axis_mask_)
      .build()
  return Tensor(op, 0, input.dtype)
}

fun TF.gather(params: Tensor, indices: Tensor, axis: Int = 0, name: String = "GatherV2"): Tensor {
  if (axis == 0) {
  
  }
  //TODO detect resource variables
  val op = g.nodeBuilder("GatherV2", ctx.getUniqueFullName(name))
      .addInput(params)
      .addInput(indices)
      .addInput(const(axis))
      .build()
  return Tensor(op, 0, params.dtype)
}

fun TF.rank(input: Tensor, name: String = "Rank", optimize: Boolean = true): Tensor {
  //TODO SparseTensor
  val input_shape = input.shape
  if (optimize && input_shape.is_fully_defined)
    return const(input_shape.rank(), name)
  return unaryOp("Rank", input, name, DT_INT32)
}