package wumo.sim.algorithm.drl.deepq

import okio.*
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.ops.assign
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.ops.group
import wumo.sim.util.ndarray.*
import wumo.sim.util.tuple2
import wumo.sim.util.tuple3
import java.io.*

fun BufferedSink.encode(act: ActFunction, prefix: String) {
  with(act.act as FunctionTensor) {
    writeInt(inputs.size)
    for (input in inputs) {
      val name = "$prefix/" + when (input) {
        is TfInput -> input.name
        is Tensor -> input.name
        else -> throw Exception()
      }
      encode(name)
    }
    writeInt(outputs.size)
    for (input in outputs) {
      val name = "$prefix/" + input.name
      encode(name)
    }
    writeInt(updates.size)
    for (input in updates) {
      val name = "$prefix/" + when (input) {
        is Operation -> input.name
        is Tensor -> input.op!!.name
        else -> throw Exception()
      }
      encode(name)
    }
    writeInt(givens.size)
    for ((t, value) in givens) {
      val name = "$prefix/" + t.name
      encode(name)
      encode(value)
    }
  }
}

fun BufferedSource.decodeActFunction(): ActFunction {
  val inputs = Array(readInt()) {
    decodeString()
  }
  val outputs = Array(readInt()) {
    decodeString()
  }
  val updates = Array(readInt()) {
    decodeString()
  }
  val givens = Array(readInt()) {
    decodeString() to decodeNDArray()
  }
  return ActFunction(function(inputs, outputs, updates, givens))
}

fun saveModel(model_file_path: String,
              act_graph_def: ByteArray,
              act_vars: List<Pair<String, NDArray<Any>>>,
              act: ActFunction) {
  File(model_file_path).sink().buffer().use { sink ->
    val prefix = "save"
    val _tf = TF()
    _tf.g.import(act_graph_def, prefix)
    defaut(_tf) {
      val init_ops = arrayListOf<Operation>()
      for ((v, value) in act_vars)
        init_ops += tf.assign(Tensor(tf.g.operation("$prefix/$v"), 0), tf.const(value.copy())).op!!
      tf.group(init_ops, name = "init")
      val bytes = tf.g.toGraphDef()
      sink.encode(bytes)
      sink.encode(act, prefix)
    }
  }
}

fun loadModel(model_file_path: String): tuple3<TF, Operation, ActFunction> {
  File(model_file_path).source().buffer().use { source ->
    val def = source.decodeByteArray()
    val act = source.decodeActFunction()
    val tf = TF()
    tf.g.import(def)
    val init = tf.g.operation("init")
    return tuple3(tf, init, act)
  }
}


