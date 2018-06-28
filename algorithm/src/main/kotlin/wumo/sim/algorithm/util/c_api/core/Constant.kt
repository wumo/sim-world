package wumo.sim.algorithm.util.c_api.core

import org.bytedeco.javacpp.tensorflow
import org.tensorflow.framework.DataType
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.c_api.DataTypefromClass
import wumo.sim.algorithm.util.c_api.Operation
import wumo.sim.algorithm.util.c_api.TF_C
import wumo.sim.algorithm.util.c_api.Tensor

fun TF_C.const(value: Any, name: String = "const"): Operation {
  Tensor.create(value).use {
    scope(name) {
      return g.opBuilder("Const", contextPath)
          .setAttr("dtype", it.dtype)
          .setAttr("value", it)
          .build()
    }
  }
}

fun TF_C.const(shape: Dimension, value: Any, name: String = "const"): Operation {
  val dtype = DataTypefromClass(value::class.java)
  tensorflow.AttrValue().use {
    it.mutable_tensor().apply {
      set_dtype(dtype.number)
      mutable_tensor_shape().apply {
        for (d in shape.elements)
          add_dim().set_size(d)
      }
      when (dtype) {
        DataType.DT_FLOAT -> add_float_val(value as Float)
        DataType.DT_DOUBLE -> add_double_val(value as Double)
        DataType.DT_INT32 -> add_int_val(value as Int)
        DataType.DT_UINT8 -> add_int_val((value as Byte).toInt())
        DataType.DT_STRING -> add_string_val(value as String)
        DataType.DT_INT64 -> add_int64_val(value as Long)
        DataType.DT_BOOL -> add_bool_val(value as Boolean)
        else -> throw IllegalArgumentException("${dtype.name} not supported")
      }
    }
    scope(name) {
      return g.opBuilder("Const", contextPath)
          .setAttr("dtype", dtype)
          .setAttr("value", it)
          .build()
    }
  }
}