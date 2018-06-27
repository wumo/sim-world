package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow
import org.tensorflow.framework.GraphDef
import wumo.sim.algorithm.util.Dimension

class TF_C {
  val g = Graph()
  val init_ops = mutableListOf<Operation>()
  
  fun placeholder(shape: Dimension, name: String): Operation {
    return g.opBuilder("Placeholder", name)
        .setAttr("dtype", DataType.FLOAT)
        .setAttr("shape", shape)
        .build()
  }
  
  fun const(shape: Dimension, value: Any, name: String): Operation {
    val dtype = DataType.fromClass(value::class.java)
    tensorflow.AttrValue().use {
      it.mutable_tensor().apply {
        set_dtype(dtype.c())
        mutable_tensor_shape().apply {
          for (d in shape.elements)
            add_dim().set_size(d)
        }
        when (dtype) {
          DataType.FLOAT -> add_float_val(value as Float)
          DataType.DOUBLE -> add_double_val(value as Double)
          DataType.INT32 -> add_int_val(value as Int)
          DataType.UINT8 -> add_int_val((value as Byte).toInt())
          DataType.STRING -> add_string_val(value as String)
          DataType.INT64 -> add_int64_val(value as Long)
          DataType.BOOL -> add_bool_val(value as Boolean)
        }
      }
      return g.opBuilder("Const", name)
          .setAttr("dtype", dtype)
          .setAttr("value", it)
          .build()
    }
  }
  
  fun variable(shape: Dimension, initial_value: Any, name: String): Operation {
    val dtype = DataType.fromClass(initial_value::class.java)
    return g.opBuilder("VariableV2", name)
        .setAttr("dtype", dtype)
        .setAttr("shape", shape)
        .build().apply {
          val initializer_const = const(shape, initial_value, "$name/initializer/const")
          init_ops += g.opBuilder("Assign", "$name/initializer")
              .addInput(this[0])
              .addInput(initializer_const[0])
              .build()
        }
  }
  
  fun variable(shape: Dimension, dtype: DataType, initializer: Operation, name: String): Operation {
    return g.opBuilder("VariableV2", name)
        .setAttr("dtype", dtype)
        .setAttr("shape", shape)
        .build().apply {
          init_ops += g.opBuilder("Assign", "$name/initializer")
              .addInput(this[0])
              .addInput(initializer[0])
              .build()
        }
  }
  
  fun global_variable_initializer(): Operation {
    return g.opBuilder("NoOp", "init")
        .apply {
          for (init_op in init_ops) {
            addControlInput(init_op)
          }
        }.build()
  }
  
  fun session(block: Session.() -> Unit) {
    Session(g).use {
      block(it)
    }
    g.close()
  }
  
  fun debugString() = GraphDef.parseFrom(g.toGraphDef()).toString()
}

internal fun throwExceptionIfNotOk(status: tensorflow.TF_Status) {
  val code = tensorflow.TF_GetCode(status)
  if (code == tensorflow.TF_OK) return
  val msg = tensorflow.TF_Message(status).string
  throw when (code) {
    tensorflow.TF_INVALID_ARGUMENT -> IllegalArgumentException(msg)
    tensorflow.TF_UNAUTHENTICATED, tensorflow.TF_PERMISSION_DENIED -> SecurityException(msg)
    tensorflow.TF_RESOURCE_EXHAUSTED, tensorflow.TF_FAILED_PRECONDITION -> IllegalStateException(msg)
    tensorflow.TF_OUT_OF_RANGE -> IndexOutOfBoundsException(msg)
    tensorflow.TF_UNIMPLEMENTED -> UnsupportedOperationException(msg)
    else -> Exception(msg)
  }
}