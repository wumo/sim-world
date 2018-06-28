package wumo.sim.algorithm.util.cpp_api.ops


import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import wumo.sim.algorithm.util.cpp_api.core.dtypeFromClass

fun TF_CPP.const(value: Any, name: String = ""): Output {
  val s = if (name.isEmpty()) scope else scope.WithOpName(name)
  var c: Class<*> = value.javaClass
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
  val tensorShapeProto = TensorShapeProto()
  for (d in shape)
    tensorShapeProto.add_dim().set_size(d)
  var c = value::class.java
  if (c.isPrimitive) {
    val dtype = dtypeFromClass(value::class.java)
    val tensorProto = TensorProto()
    tensorProto.set_dtype(dtype)
    tensorProto.set_allocated_tensor_shape(tensorShapeProto)
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
  val s = TensorShape(tensorShapeProto)
  c = c.componentType
  if (!c.isPrimitive)
    throw IllegalArgumentException("Only support primitive array")
  val tensor = when (c) {
    Float::class.java -> Tensor.create(value as FloatArray, s)
    Double::class.java -> Tensor.create(value as DoubleArray, s)
    Int::class.java -> Tensor.create(value as IntArray, s)
    Long::class.java -> Tensor.create(value as LongArray, s)
    Byte::class.java -> Tensor.create(value as ByteArray, s)
    Short::class.java -> Tensor.create(value as ShortArray, s)
    String::class.java -> Tensor.create(value as Array<out String>, s)
    else -> throw IllegalArgumentException("Not supported: $c")
  }
  return Const(scope.WithOpName(name), tensor)
}