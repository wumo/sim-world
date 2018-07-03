package wumo.sim.algorithm.util.cpp_api.ops

import org.apache.commons.lang3.ClassUtils.isPrimitiveWrapper
import org.apache.commons.lang3.ClassUtils.wrapperToPrimitive
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import wumo.sim.algorithm.util.cpp_api.dtypeFromClass
import wumo.sim.algorithm.util.dim
import java.lang.reflect.Array

fun TF_CPP.const(value: Float, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)
fun TF_CPP.const(value: Double, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)
fun TF_CPP.const(value: Boolean, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)
fun TF_CPP.const(value: Byte, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)
fun TF_CPP.const(value: Short, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)
fun TF_CPP.const(value: Int, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)
fun TF_CPP.const(value: Long, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)
fun TF_CPP.const(value: String, name: String = "", scope: Scope = root) = Const(scope.WithOpName(name), value)

fun TF_CPP.const1D(vararg value: Float, name: String = "", scope: Scope = root) =
    const(dim(value.size), value, name, scope)

fun TF_CPP.const1D(vararg value: Double, name: String = "", scope: Scope = root) =
    const(dim(value.size), value, name, scope)

fun TF_CPP.const1D(vararg value: Int, name: String = "", scope: Scope = root) =
    const(dim(value.size), value, name, scope)

fun TF_CPP.const1D(vararg value: Long, name: String = "", scope: Scope = root) =
    const(dim(value.size), value, name, scope)

fun TF_CPP.const1D(vararg value: Byte, name: String = "", scope: Scope = root) =
    const(dim(value.size), value, name, scope)

fun TF_CPP.const1D(vararg value: Short, name: String = "", scope: Scope = root) =
    const(dim(value.size), value, name, scope)

fun TF_CPP.const1D(vararg value: String, name: String = "", scope: Scope = root) =
    const(dim(value.size), value, name, scope)

fun TF_CPP.const(shape: Dimension, value: Any, name: String = "", scope: Scope = root): Output {
  val tensorShapeProto = TensorShapeProto()
  for (d in shape)
    tensorShapeProto.add_dim().set_size(d)
  var c = value::class.java
  if (c.isArray) {
    val s = TensorShape(tensorShapeProto)
    c = c.componentType
    var wrapper = false
    c = when {
      c.isPrimitive -> c
      isPrimitiveWrapper(c) -> {
        wrapper = true
        wrapperToPrimitive(c)
      }
      c == String::class.java -> c
      else -> throw IllegalArgumentException("Only support primitive array")
    }
    val tensor = when (c) {
      Float::class.java -> Tensor.create(
          if (wrapper)
            FloatArray(Array.getLength(value)) { Array.get(value, it) as Float }
          else value as FloatArray, s)
      Double::class.java -> Tensor.create(
          if (wrapper)
            DoubleArray(Array.getLength(value)) { Array.get(value, it) as Double }
          else value as DoubleArray, s)
      Boolean::class.java -> Tensor.create(ByteArray(Array.getLength(value)) {
        val element = Array.get(value, it) as Boolean
        if (element) 1 else 0
      }, s)
      Byte::class.java -> Tensor.create(
          if (wrapper)
            ByteArray(Array.getLength(value)) { Array.get(value, it) as Byte }
          else value as ByteArray, s)
      Short::class.java -> Tensor.create(
          if (wrapper)
            ShortArray(Array.getLength(value)) { Array.get(value, it) as Short }
          else value as ShortArray, s)
      Int::class.java -> Tensor.create(
          if (wrapper)
            IntArray(Array.getLength(value)) { Array.get(value, it) as Int }
          else value as IntArray, s)
      Long::class.java -> Tensor.create(
          if (wrapper)
            LongArray(Array.getLength(value)) { Array.get(value, it) as Long }
          else value as LongArray, s)
      String::class.java -> Tensor.create(value as kotlin.Array<out String>, s)
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
    is Boolean -> tensorProto.add_bool_val(value)
    is Byte -> tensorProto.add_int_val(value.toInt())
    is Short -> tensorProto.add_int_val(value.toInt())
    is Int -> tensorProto.add_int_val(value)
    is Long -> tensorProto.add_int64_val(value)
    is String -> tensorProto.add_string_val(value)
    else -> throw IllegalArgumentException("Not supported: ${value::class}")
  }
  return ConstFromProto(scope.WithOpName(name), tensorProto)
}