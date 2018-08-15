package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.AttrValue
import org.bytedeco.javacpp.tensorflow.TensorProto
import wumo.sim.tensorflow.Tensor
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.dtypeFromClass
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape
import wumo.sim.util.SwitchValue3
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.scalarDimension
import wumo.sim.util.toByte

object const_ops {
  val const_switch = SwitchValue3<DataType<*>, Shape, Any, String, Output>().apply {
    case(INT8, UINT8) { tf.const(_1, (_2 as Number).toByte(), _3) }
    case(FLOAT) { tf.const(_1, (_2 as Number).toFloat(), _3) }
    case(DOUBLE) { tf.const(_1, (_2 as Number).toDouble(), _3) }
    case(BOOL) { tf.const(_1, (_2 as Number) != 0, _3) }
    case(INT16, UINT16) { tf.const(_1, (_2 as Number).toShort(), _3) }
    case(INT32, UINT32) { tf.const(_1, (_2 as Number).toInt(), _3) }
    case(INT64, UINT64) { tf.const(_1, (_2 as Number).toLong(), _3) }
    case(STRING) { tf.const(_1, _2.toString(), _3) }
  }
  
  interface API {
    fun const(value: Float, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: Double, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: Boolean, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: Byte, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: Short, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: Int, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: Long, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: String, name: String = "Const") = const(scalarDimension, value, name)
    fun const(value: FloatArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
    fun const(value: Array<Float>, name: String = "Const") = const(Tensor(Shape(value.size), value.toFloatArray()), name)
    fun const(value: DoubleArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
    fun const(value: Array<Double>, name: String = "Const") = const(Tensor(Shape(value.size), value.toDoubleArray()), name)
    fun const(value: BooleanArray, name: String = "Const") = const(Tensor(Shape(value.size), ByteArray(value.size) { value[it].toByte() }), name)
    fun const(value: Array<Boolean>, name: String = "Const") = const(Tensor(Shape(value.size), value.toBooleanArray()), name)
    fun const(value: ByteArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
    fun const(value: Array<Byte>, name: String = "Const") = const(Tensor(Shape(value.size), value.toByteArray()), name)
    fun const(value: ShortArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
    fun const(value: Array<Short>, name: String = "Const") = const(Tensor(Shape(value.size), value.toShortArray()), name)
    fun const(value: IntArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
    fun const(value: Array<Int>, name: String = "Const") = const(Tensor(Shape(value.size), value.toIntArray()), name)
    fun const(value: LongArray, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
    fun const(value: Array<Long>, name: String = "Const") = const(Tensor(Shape(value.size), value.toLongArray()), name)
    fun const(value: Array<String>, name: String = "Const") = const(Tensor(Shape(value.size), value), name)
    fun const(shape: Shape, value: Float, name: String = "Const") = const(shape, FLOAT, name) { add_float_val(value) }
    fun const(shape: Shape, value: Double, name: String = "Const") = const(shape, DOUBLE, name) { add_double_val(value) }
    fun const(shape: Shape, value: Boolean, name: String = "Const") = const(shape, BOOL, name) { add_bool_val(value) }
    fun const(shape: Shape, value: Byte, name: String = "Const") = const(shape, INT8, name) { add_int_val(value.toInt()) }
    fun const(shape: Shape, value: Short, name: String = "Const") = const(shape, INT16, name) { add_int_val(value.toInt()) }
    fun const(shape: Shape, value: Int, name: String = "Const") = const(shape, INT32, name) { add_int_val(value) }
    fun const(shape: Shape, value: Long, name: String = "Const") = const(shape, INT64, name) { add_int64_val(value) }
    fun const(shape: Shape, value: String, name: String = "Const") = const(shape, STRING, name) { add_string_val(value) }
    fun const(shape: Shape, value: FloatArray, name: String = "Const") = const(Tensor(shape, value), name)
    fun const(shape: Shape, value: DoubleArray, name: String = "Const") = const(Tensor(shape, value), name)
    fun const(shape: Shape, value: BooleanArray, name: String = "Const") = const(Tensor(shape, ByteArray(value.size) { value[it].toByte() }), name)
    fun const(shape: Shape, value: ByteArray, name: String = "Const") = const(Tensor(shape, value), name)
    fun const(shape: Shape, value: ShortArray, name: String = "Const") = const(Tensor(shape, value), name)
    fun const(shape: Shape, value: IntArray, name: String = "Const") = const(Tensor(shape, value), name)
    fun const(shape: Shape, value: LongArray, name: String = "Const") = const(Tensor(shape, value), name)
    fun const(shape: Shape, value: Array<String>, name: String = "Const") = const(Tensor(shape, value), name)
    fun <R> const(dtype: DataType<R>, value: Any, name: String = "Const") =
        const(scalarDimension, dtype, value, name)
    
    fun const(shape: Shape, dtype: DataType<*>, value: Any, name: String = "Const"): Output {
      return const_switch(dtype, shape, value, name)
    }
    
    private fun const(shape: Shape, dtype: DataType<*>, name: String = "Const", set_value: TensorProto.() -> Unit): Output {
      val tensor_proto = AttrValue()
      tensor_proto.mutable_tensor().apply {
        set_dtype(dtype.cValue)
        mutable_tensor_shape().apply {
          for (d in shape)
            add_dim().set_size(d.toLong())
        }
        set_value(this)
      }
      return buildOpTensor("Const", name = name) {
        attr("value", tensor_proto)
        attr("dtype", dtype)
      }
    }
    
    fun <T : Any> const(value: Tensor<T>, name: String = "Const") =
        buildOpTensor("Const", name = name) {
          attr("value", value)
          attr("dtype", value.dtype.base_dtype.cValue)
        }
    
    fun <T : Any> const(value: NDArray<T>, name: String = "Const"): Output {
      val dtype = dtypeFromClass(value.dtype)
      return buildOpTensor("Const", name = name) {
        attr("value", Tensor.fromNDArray(value))
        attr("dtype", dtype)
      }
    }
  }
}