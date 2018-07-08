package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.TensorValue
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.dim
import wumo.sim.algorithm.util.helpers.toByte
import wumo.sim.algorithm.util.scalarDimension

fun TF.const(value: Float, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Double, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Boolean, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Byte, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Short, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Int, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Long, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: String, name: String = "Const") = const(scalarDimension, value, name)

fun TF.const(value: FloatArray, name: String = "Const") = const(TensorValue(dim(value.size), value), name)
fun TF.const(value: DoubleArray, name: String = "Const") = const(TensorValue(dim(value.size), value), name)
fun TF.const(value: BooleanArray, name: String = "Const") = const(TensorValue(dim(value.size), ByteArray(value.size) { value[it].toByte() }), name)
fun TF.const(value: ByteArray, name: String = "Const") = const(TensorValue(dim(value.size), value), name)
fun TF.const(value: ShortArray, name: String = "Const") = const(TensorValue(dim(value.size), value), name)
fun TF.const(value: IntArray, name: String = "Const") = const(TensorValue(dim(value.size), value), name)
fun TF.const(value: LongArray, name: String = "Const") = const(TensorValue(dim(value.size), value), name)
fun TF.const(value: Array<String>, name: String = "Const") = const(TensorValue(dim(value.size), value), name)

fun TF.const(shape: Dimension, value: Float, name: String = "Const") = const(shape, DT_FLOAT, name) { add_float_val(value) }
fun TF.const(shape: Dimension, value: Double, name: String = "Const") = const(shape, DT_DOUBLE, name) { add_double_val(value) }
fun TF.const(shape: Dimension, value: Boolean, name: String = "Const") = const(shape, DT_BOOL, name) { add_bool_val(value) }
fun TF.const(shape: Dimension, value: Byte, name: String = "Const") = const(shape, DT_INT8, name) { add_int_val(value.toInt()) }
fun TF.const(shape: Dimension, value: Short, name: String = "Const") = const(shape, DT_INT16, name) { add_int_val(value.toInt()) }
fun TF.const(shape: Dimension, value: Int, name: String = "Const") = const(shape, DT_INT32, name) { add_int_val(value) }
fun TF.const(shape: Dimension, value: Long, name: String = "Const") = const(shape, DT_INT64, name) { add_int64_val(value) }
fun TF.const(shape: Dimension, value: String, name: String = "Const") = const(shape, DT_STRING, name) { add_string_val(value) }

fun TF.const(shape: Dimension, value: FloatArray, name: String = "Const") = const(TensorValue(shape, value), name)
fun TF.const(shape: Dimension, value: DoubleArray, name: String = "Const") = const(TensorValue(shape, value), name)
fun TF.const(shape: Dimension, value: BooleanArray, name: String = "Const") = const(TensorValue(shape, ByteArray(value.size) { value[it].toByte() }), name)
fun TF.const(shape: Dimension, value: ByteArray, name: String = "Const") = const(TensorValue(shape, value), name)
fun TF.const(shape: Dimension, value: ShortArray, name: String = "Const") = const(TensorValue(shape, value), name)
fun TF.const(shape: Dimension, value: IntArray, name: String = "Const") = const(TensorValue(shape, value), name)
fun TF.const(shape: Dimension, value: LongArray, name: String = "Const") = const(TensorValue(shape, value), name)
fun TF.const(shape: Dimension, value: Array<String>, name: String = "Const") = const(TensorValue(shape, value), name)

fun TF.const(dtype: Int, value: Any, name: String = "Const") =
    const(scalarDimension, dtype, value, name)

fun TF.const(shape: Dimension, dtype: Int, value: Any, name: String = "Const"): Tensor {
  return when (dtype) {
    DT_FLOAT -> const(shape, (value as Number).toFloat(), name)
    DT_DOUBLE -> const(shape, (value as Number).toDouble(), name)
    DT_BOOL -> const(shape, (value as Number) != 0, name)
    DT_INT8, DT_UINT8 -> const(shape, (value as Number).toByte(), name)
    DT_INT16, DT_UINT16 -> const(shape, (value as Number).toShort(), name)
    DT_INT32, DT_UINT32 -> const(shape, (value as Number).toInt(), name)
    DT_INT64, DT_UINT64 -> const(shape, (value as Number).toLong(), name)
    DT_STRING -> const(shape, value.toString(), name)
    else -> throw IllegalArgumentException("unsupported type $dtype")
  }
}

private fun TF.const(shape: Dimension, dtype: Int, name: String = "Const", set_value: TensorProto.() -> Unit): Tensor {
  val tensor_proto = AttrValue()
  tensor_proto.mutable_tensor().apply {
    set_dtype(dtype)
    mutable_tensor_shape().apply {
      for (d in shape)
        add_dim().set_size(d.toLong())
    }
    set_value(this)
  }
  
  val op = g.nodeBuilder("Const", ctx.getUniqueFullName(name))
      .setAttr("value", tensor_proto)
      .setAttrType("dtype", dtype)
      .build()
  return Tensor(op, 0, dtype)
}

fun <T> TF.const(value: TensorValue<T>, name: String = "Const"): Tensor {
  val op = g.nodeBuilder("Const", ctx.getUniqueFullName(name))
      .setAttr("value", value)
      .setAttrType("dtype", value.dtype)
      .build()
  return Tensor(op, 0, value.dtype)
}