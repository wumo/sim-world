package wumo.sim.algorithm.util.java_api

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import java.nio.*

class TFJava {
  val g = Graph()
  val init_ops = mutableListOf<Operation>()
  
  fun placeholder(shape: Dimension, name: String): Operation {
    return g.opBuilder("Placeholder", name)
        .setAttr("dtype", DataType.FLOAT)
        .setAttr("shape", Shape.make(shape.firstDim.toLong(), *shape.otherDim))
        .build()
  }
  
  inline fun <reified T> variable(shape: Dimension, initial_value: T, name: String): Operation {
    val n = shape.numElements().toInt()
    val value = when (initial_value) {
      is Float -> Tensor.create(shape.asLongArray(), FloatBuffer.allocate(n).apply { repeat(n) { put(initial_value) };flip() })
      is Double -> Tensor.create(shape.asLongArray(), DoubleBuffer.allocate(n).apply { repeat(n) { put(initial_value) };flip() })
      is Int -> Tensor.create(shape.asLongArray(), IntBuffer.allocate(n).apply { repeat(n) { put(initial_value) };flip() })
      is Long -> Tensor.create(shape.asLongArray(), LongBuffer.allocate(n).apply { repeat(n) { put(initial_value) };flip() })
      else -> throw IllegalArgumentException("DataType ${T::class.java} is not supported yet")
    }
    val dtype = DataType.fromClass(T::class.java)
    val initializer = g.opBuilder("Const", "$name/initial_value")
        .setAttr("dtype", dtype)
        .setAttr("value", value)
        .build()
    return variable<T>(shape, initializer, name)
  }
  
  inline fun <reified T> variable(shape: Dimension, initializer: Operation, name: String): Operation {
    val dtype = DataType.fromClass(T::class.java)
    val v = g.opBuilder("VariableV2", name)
        .setAttr("dtype", dtype)
        .setAttr("shape", Shape.make(shape.firstDim.toLong(), *shape.otherDim))
        .build()
    val assign = g.opBuilder("Assign", "$name/assign")
        .addInput(v.output<T>(0))
        .addInput(initializer.output<T>(0))
        .build()
    init_ops += assign
    return v
  }
  
  inline fun <reified T> const(value: T, name: String): Operation {
    val value = Tensor.create(value)
    return g.opBuilder("Const", name)
        .setAttr("dtype", value.dataType())
        .setAttr("value", value)
        .build()
  }
  
  fun writeTextProto(path: String) {
    val def = tensorflow.GraphDef()
    assert(def.ParseFromString(BytePointer(*g.toGraphDef())))
    tensorflow.TF_CHECK_OK(tensorflow.WriteTextProto(tensorflow.Env.Default(), path, def))
  }
  
  fun session(block: Session.() -> Unit) {
    val session = Session(g)
    block(session)
    session.close()
    g.close()
  }
  
  fun global_variable_initializer(): Operation {
    return g.opBuilder("NoOp", "init").apply {
      for (init_op in init_ops)
        addControlInput(init_op)
    }.build()
  }
}