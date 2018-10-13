package wumo.sim.algorithm.drl.deepq

import okio.BufferedSink
import okio.BufferedSource
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.allocateTensor
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.memcpy
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.drl.common.TFFunctionTensor
import wumo.sim.algorithm.drl.common.functionFromName
import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tensor.Tensor
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.toDataType
import wumo.sim.util.ndarray.*
import wumo.sim.util.sink
import wumo.sim.util.source
import wumo.sim.util.t2
import wumo.sim.util.t3
import java.io.File
import java.nio.ByteBuffer

fun BufferedSink.encode(act: ActFunction, prefix: String = "") {
  val prefix = if (prefix.isBlank()) "" else "$prefix/"
  with(act.act as TFFunctionTensor) {
    writeInt(inputs.size)
    for (input in inputs) {
      val name = prefix + when (input) {
        is TfInput -> input.name
        is Output -> input.name
        else -> throw Exception()
      }
      encode(name)
    }
    writeInt(outputs.size)
    for (input in outputs) {
      val name = prefix + input.name
      encode(name)
    }
    writeInt(updates.size)
    for (input in updates) {
      val name = prefix + when (input) {
        is Op -> input.name
        is Output -> input.op.name
        else -> throw Exception()
      }
      encode(name)
    }
    writeInt(givens.size)
    for ((t, value) in givens) {
      val name = prefix + t.name
      encode(name)
      encode(value)
    }
  }
}

fun BufferedSource.decodeActFunction(): ActFunction {
  val inputs = List(readInt()) {
    decodeString()
  }
  val outputs = List(readInt()) {
    decodeString()
  }
  val updates = List(readInt()) {
    decodeString()
  }
  val givens = List(readInt()) {
    decodeString() to decodeNDArray()
  }
  return ActFunction(functionFromName(inputs, outputs, updates, givens))
}

fun saveVariable(act_vars: List<Pair<String, NDArray<Any>>>) {
  File(System.getProperty("java.io.tmpdir")
           + File.separatorChar + "model").sink { sink ->
    sink.encode(act_vars.size)
    for ((v, value) in act_vars) {
      val t = Tensor.fromNDArray(value)
      val c_tensor = t.c_tensor
      val src = TF_TensorData(c_tensor)
      val size = TF_TensorByteSize(c_tensor)
      val data = BytePointer(src)
      data.capacity(size)
      val buffer = data.asByteBuffer()
      
      sink.encode(v)
      sink.writeInt(value.dtype.toDataType().cValue)
      sink.encode(value.shape.asLongArray()!!)
      sink.writeLong(size)
      sink.write(buffer)
      c_tensor.deallocate()
    }
  }
}

fun loadVariable(): List<Pair<String, Output>> {
  val act_vars = mutableListOf<Pair<String, Output>>()
  File(System.getProperty("java.io.tmpdir")
           + File.separatorChar + "model").source { source ->
    val totalSize = source.decodeInt()
    for (i in 0 until totalSize) {
      val name = source.decodeString()
      val dtype = source.readInt().toDataType<Any>()
      val dims = source.decodeLongArray()
      val size = source.readLong()
      val buf = ByteBuffer.allocate(size.toInt())
      val readSize = source.read(buf)
      assert(size.toInt() == readSize)
      val src = BytePointer(buf)
      val t = allocateTensor(dtype.cValue, dims, size)
      val data = TF_TensorData(t)
      memcpy(data, src, size)
      act_vars += name to tf.const(t, dtype)
    }
  }
  return act_vars
}

fun saveModel(model_file_path: String,
              build_act: () -> t2<ActFunction, Set<Variable>>,
              act_vars: List<Pair<String, NDArray<Any>>>) {
  File(model_file_path).sink { sink ->
    val graph = Graph()
    tf.unsafeDefaultGraph(graph) {
      val (act, act_v) = build_act()
      val act_v_str = act_v.mapTo(mutableSetOf()) { it.name }
      val init_ops = mutableListOf<Op>()
      for ((v, value) in act_vars.filter { it.first in act_v_str })
        init_ops += tf.assign(tf.currentGraph.getTensor(v),
                              tf.const(value.copy())).op
      tf.group(init_ops, name = "init")
      val bytes = tf.currentGraph.toGraphDefBytes()
      sink.encode(bytes)
      sink.encode(act)
    }
  }
}

fun loadModel(model_file_path: String): t3<Graph, Op, ActFunction> {
  File(model_file_path).source { source ->
    val def = source.decodeByteArray()
    val act = source.decodeActFunction()
    val graph = Graph()
    val def_ptr = BytePointer(*def)
    graph.import(def_ptr)
    val init = graph.findOp("init")!!
    return t3(graph, init, act)
  }
}


