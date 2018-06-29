package wumo.sim.algorithm.util.cpp_api.ops


import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import wumo.sim.algorithm.util.cpp_api.dtypeFromClass

fun TF_CPP.const(value: Any, name: String = "", scope: Scope = root): Output {
  val s = scope.WithOpName(name)
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

fun TF_CPP.const(shape: Dimension, value: Any, name: String = "", scope: Scope = root): Output {
  val tensorShapeProto = TensorShapeProto()
  for (d in shape)
    tensorShapeProto.add_dim().set_size(d)
  var c = value::class.java
  if (c.isArray) {
    val s = TensorShape(tensorShapeProto)
    c = c.componentType
    c = when {
      c.isPrimitive -> c
      else -> throw IllegalArgumentException("Only support primitive(except wrapper) array")
    }
    val tensor = when (c) {
      Float::class.java -> Tensor.create(value as FloatArray, s)
      Double::class.java -> Tensor.create(value as DoubleArray, s)
      Int::class.java -> Tensor.create(value as IntArray, s)
      Long::class.java -> Tensor.create(value as LongArray, s)
      Byte::class.java -> Tensor.create(value as ByteArray, s)
      Short::class.java -> Tensor.create(value as ShortArray, s)
      else -> throw IllegalArgumentException("Not supported: $c")
    }
    return Const(scope.WithOpName(name), tensor)
  }
  val dtype = dtypeFromClass(value::class.java)
  val tensorProto = TensorProto()
  tensorProto.set_dtype(dtype)
  tensorProto.mutable_tensor_shape().CopyFrom(tensorShapeProto)
  when (value) {
    is Float -> tensorProto.add_float_val(value)
    is Double -> tensorProto.add_double_val(value)
    is Int -> tensorProto.add_int_val(value)
    is Long -> tensorProto.add_int64_val(value)
    is Byte -> tensorProto.add_int_val(value.toInt())
    is Boolean -> tensorProto.add_bool_val(value)
    else -> throw IllegalArgumentException("Not supported: ${value::class}")
  }
  return ConstFromProto(scope.WithOpName(name), tensorProto)
}