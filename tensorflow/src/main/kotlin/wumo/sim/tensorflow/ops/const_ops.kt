package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.*
import wumo.sim.tensorflow.Tensor
import wumo.sim.util.Shape
import wumo.sim.util.SwitchValue3
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.scalarDimension
import wumo.sim.util.toByte

fun TF.const(value: Float, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Double, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Boolean, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Byte, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Short, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Int, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: Long, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: String, name: String = "Const") = const(scalarDimension, value, name)
fun TF.const(value: FloatArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
fun TF.const(value: Array<Float>, name: String = "Const") = const(Tensor(Shape(value.size), value.toFloatArray()), name)
fun TF.const(value: DoubleArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
fun TF.const(value: Array<Double>, name: String = "Const") = const(Tensor(Shape(value.size), value.toDoubleArray()), name)
fun TF.const(value: BooleanArray, name: String = "Const") = const(Tensor(Shape(value.size), ByteArray(value.size) { value[it].toByte() }), name)
fun TF.const(value: Array<Boolean>, name: String = "Const") = const(Tensor(Shape(value.size), value.toBooleanArray()), name)
fun TF.const(value: ByteArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
fun TF.const(value: Array<Byte>, name: String = "Const") = const(Tensor(Shape(value.size), value.toByteArray()), name)
fun TF.const(value: ShortArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
fun TF.const(value: Array<Short>, name: String = "Const") = const(Tensor(Shape(value.size), value.toShortArray()), name)
fun TF.const(value: IntArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
fun TF.const(value: Array<Int>, name: String = "Const") = const(Tensor(Shape(value.size), value.toIntArray()), name)
fun TF.const(value: LongArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
fun TF.const(value: Array<Long>, name: String = "Const") = const(Tensor(Shape(value.size), value.toLongArray()), name)
fun TF.const(value: Array<String>, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
fun TF.const(shape: Shape, value: Float, name: String = "Const") = const(shape, DT_FLOAT, name) { add_float_val(value) }
fun TF.const(shape: Shape, value: Double, name: String = "Const") = const(shape, DT_DOUBLE, name) { add_double_val(value) }
fun TF.const(shape: Shape, value: Boolean, name: String = "Const") = const(shape, DT_BOOL, name) { add_bool_val(value) }
fun TF.const(shape: Shape, value: Byte, name: String = "Const") = const(shape, DT_INT8, name) { add_int_val(value.toInt()) }
fun TF.const(shape: Shape, value: Short, name: String = "Const") = const(shape, DT_INT16, name) { add_int_val(value.toInt()) }
fun TF.const(shape: Shape, value: Int, name: String = "Const") = const(shape, DT_INT32, name) { add_int_val(value) }
fun TF.const(shape: Shape, value: Long, name: String = "Const") = const(shape, DT_INT64, name) { add_int64_val(value) }
fun TF.const(shape: Shape, value: String, name: String = "Const") = const(shape, DT_STRING, name) { add_string_val(value) }
fun TF.const(shape: Shape, value: FloatArray, name: String = "Const") = const(Tensor(shape, value), name)
fun TF.const(shape: Shape, value: DoubleArray, name: String = "Const") = const(Tensor(shape, value), name)
fun TF.const(shape: Shape, value: BooleanArray, name: String = "Const") = const(Tensor(shape, ByteArray(value.size) { value[it].toByte() }), name)
fun TF.const(shape: Shape, value: ByteArray, name: String = "Const") = const(Tensor(shape, value), name)
fun TF.const(shape: Shape, value: ShortArray, name: String = "Const") = const(Tensor(shape, value), name)
fun TF.const(shape: Shape, value: IntArray, name: String = "Const") = const(Tensor(shape, value), name)
fun TF.const(shape: Shape, value: LongArray, name: String = "Const") = const(Tensor(shape, value), name)
fun TF.const(shape: Shape, value: Array<String>, name: String = "Const") = const(Tensor(shape, value), name)
fun TF.const(dtype: Int, value: Any, name: String = "Const") =
    const(scalarDimension, dtype, value, name)

val const_switch = SwitchValue3<Int, Shape, Any, String, Output>().apply {
  case(DT_INT8, DT_UINT8) { tf.const(_1, (_2 as Number).toByte(), _3) }
  case(DT_FLOAT) { tf.const(_1, (_2 as Number).toFloat(), _3) }
  case(DT_DOUBLE) { tf.const(_1, (_2 as Number).toDouble(), _3) }
  case(DT_BOOL) { tf.const(_1, (_2 as Number) != 0, _3) }
  case(DT_INT16, DT_UINT16) { tf.const(_1, (_2 as Number).toShort(), _3) }
  case(DT_INT32, DT_UINT32) { tf.const(_1, (_2 as Number).toInt(), _3) }
  case(DT_INT64, DT_UINT64) { tf.const(_1, (_2 as Number).toLong(), _3) }
  case(DT_STRING) { tf.const(_1, _2.toString(), _3) }
}

fun TF.const(shape: Shape, dtype: Int, value: Any, name: String = "Const"): Output {
  return const_switch(dtype, shape, value, name)
}

private fun TF.const(shape: Shape, dtype: Int, name: String = "Const", set_value: TensorProto.() -> Unit): Output {
  val tensor_proto = AttrValue()
  tensor_proto.mutable_tensor().apply {
    set_dtype(dtype)
    mutable_tensor_shape().apply {
      for (d in shape)
        add_dim().set_size(d.toLong())
    }
    set_value(this)
  }
  return buildOpTensor("Const", name = name) {
    attr("value", tensor_proto)
    attrType("dtype", dtype)
  }
}

fun <T : Any> TF.const(value: Tensor<T>, name: String = "Const") =
    buildOpTensor("Const", name = name) {
      attr("value", value)
      attrType("dtype", value.dtype.base_dtype)
    }

fun <T : Any> TF.const(value: NDArray<T>, name: String = "Const"): Output {
  val dtype = dtypeFromClass(value.dtype)
  return buildOpTensor("Const", name = name) {
    attr("value", Tensor.fromNDArray(value))
    attrType("dtype", dtype)
  }
}