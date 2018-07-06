package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Scope
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.TensorValue
import wumo.sim.algorithm.util.Dimension

private val scalarDimension = Dimension()
fun TF.const(value: Float, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
fun TF.const(value: Double, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
fun TF.const(value: Boolean, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
fun TF.const(value: Byte, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
fun TF.const(value: Short, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
fun TF.const(value: Int, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
fun TF.const(value: Long, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
fun TF.const(value: String, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)

fun TF.const(shape: Dimension, value: Float, name: String = "", scope: Scope = root) = const(shape, DT_FLOAT, name, scope) { add_float_val(value) }
fun TF.const(shape: Dimension, value: Double, name: String = "", scope: Scope = root) = const(shape, DT_DOUBLE, name, scope) { add_double_val(value) }
fun TF.const(shape: Dimension, value: Boolean, name: String = "", scope: Scope = root) = const(shape, DT_BOOL, name, scope) { add_bool_val(value) }
fun TF.const(shape: Dimension, value: Byte, name: String = "", scope: Scope = root) = const(shape, DT_INT8, name, scope) { add_int_val(value.toInt()) }
fun TF.const(shape: Dimension, value: Short, name: String = "", scope: Scope = root) = const(shape, DT_INT16, name, scope) { add_int_val(value.toInt()) }
fun TF.const(shape: Dimension, value: Int, name: String = "", scope: Scope = root) = const(shape, DT_INT32, name, scope) { add_int_val(value) }
fun TF.const(shape: Dimension, value: Long, name: String = "", scope: Scope = root) = const(shape, DT_INT64, name, scope) { add_int64_val(value) }
fun TF.const(shape: Dimension, value: String, name: String = "", scope: Scope = root) = const(shape, DT_STRING, name, scope) { add_string_val(value) }

fun TF.const(shape: Dimension, dtype: Int, name: String = "", scope: Scope = root, set_value: TensorProto.() -> Unit): Tensor {
  val tensor_proto = AttrValue()
  tensor_proto.mutable_tensor().apply {
    set_dtype(dtype)
    mutable_tensor_shape().apply {
      for (d in shape)
        add_dim().set_size(d)
    }
    set_value(this)
  }
  
  val unique_name = scope.getUniqueNameForOp(name)
  val op = g.nodeBuilder("Const", unique_name)
      .setAttr("value", tensor_proto)
      .setAttrType("dtype", dtype)
      .build()
  return Tensor(op, 0, dtype)
}

fun TF.const(value: FloatArray, name: String = "", scope: Scope = root) = const(TensorValue.create(value), name, scope)
//fun TF.const(value: DoubleArray, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
//fun TF.const(value: BooleanArray, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
//fun TF.const(value: ByteArray, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
//fun TF.const(value: ShortArray, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
//fun TF.const(value: IntArray, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
//fun TF.const(value: LongArray, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)
//fun TF.const(value: Array<String>, name: String = "", scope: Scope = root) = const(scalarDimension, value, name, scope)

fun <T> TF.const(value: TensorValue<T>, name: String = "", scope: Scope = root): Tensor {
  val unique_name = scope.getUniqueNameForOp(name)
  val op = g.nodeBuilder("Const", unique_name)
      .setAttr("value", value)
      .setAttrType("dtype", value.dtype)
      .build()
  return Tensor(op, 0, value.dtype)
}