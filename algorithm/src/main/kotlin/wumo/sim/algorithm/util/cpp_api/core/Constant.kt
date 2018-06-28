package wumo.sim.algorithm.util.cpp_api.core

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP

fun TF_CPP.const(value: Any, name: String = ""): Output {
  val s = if (name.isEmpty()) scope else scope.WithOpName(name)
  return when (value) {
    is Float -> Const(s, value)
    is Double -> Const(s, value)
    is Int -> Const(s, value)
    is Long -> Const(s, value)
    is Byte -> Const(s, value)
    is Boolean -> Const(s, value)
    is Short -> Const(s, value)
    is String -> Const(s, value)
    else -> throw IllegalArgumentException("Not supported: ${value::class}")
  }
}

fun TF_CPP.const(shape: Dimension, value: Any, name: String = ""): Output {
  val dtype = dtypeFromClass(value::class.java)
  val tensorProto = TensorProto()
  tensorProto.set_dtype(dtype)
  tensorProto.mutable_tensor_shape().apply {
    for (d in shape)
      add_dim().set_size(d)
  }
  when (value) {
    is Float -> tensorProto.add_float_val(value)
    is Double -> tensorProto.add_double_val(value)
    is Int -> tensorProto.add_int_val(value)
    is Long -> tensorProto.add_int64_val(value)
    is Byte -> tensorProto.add_int_val(value.toInt())
    is Boolean -> tensorProto.add_bool_val(value)
    else -> throw IllegalArgumentException("Not supported: ${value::class}")
  }
  return ConstFromProto(if (name.isEmpty()) scope else scope.WithOpName(name), tensorProto)
}