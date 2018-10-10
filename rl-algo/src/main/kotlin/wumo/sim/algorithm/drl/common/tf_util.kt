package wumo.sim.algorithm.drl.common

import wumo.sim.algorithm.drl.deepq.TfInput
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.minus
import wumo.sim.tensorflow.ops.basic.times
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray

interface Function {
  operator fun invoke(vararg args: Any): List<NDArray<*>>
}

class FunctionTensor(val inputs: List<Any>,
                     val outputs: List<Output>,
                     val updates: List<Any>,
                     val givens: List<Pair<Output, NDArray<*>>>) : Function {
  
  override operator fun invoke(vararg args: Any): List<NDArray<*>> {
    assert(args.size <= inputs.size) { "Too many arguments provided" }
    val feed_dict = mutableMapOf<Output, NDArray<*>>()
    for (i in 0 until inputs.size) {
      val input = inputs[i]
      val value = NDArray.toNDArray<Any>(args[i])
      feed_dict += when (input) {
        is TfInput -> input.make_feed_dict(value)
        is Output -> input to value
        else -> throw Exception()
      }
    }
    for ((input, value) in givens)
      feed_dict.putIfAbsent(input, value)
    return tf.currentSession!!.run(outputs, updates, feed_dict = feed_dict)
  }
}

class FunctionString(val inputs: List<String>,
                     val outputs: List<String>,
                     val updates: List<String>,
                     val givens: List<Pair<String, NDArray<*>>>) : Function {
  
  override operator fun invoke(vararg args: Any): List<NDArray<*>> {
    assert(args.size <= inputs.size) { "Too many arguments provided" }
    val feed_dict = mutableMapOf<String, NDArray<*>>()
    for (i in 0 until inputs.size) {
      val input = inputs[i]
      val value = NDArray.toNDArray<Any>(args[i])
      feed_dict += input to value
    }
    for ((input, value) in givens)
      feed_dict.putIfAbsent(input, value)
    return tf.currentSession!!.runString(outputs, updates, feed_dict = feed_dict)
  }
}

fun functionFromName(inputs: List<String> = emptyList(),
                     outputs: List<String> = emptyList(),
                     updates: List<String> = emptyList(),
                     givens: List<Pair<String, NDArray<*>>> = emptyList()): Function {
  return FunctionString(inputs, outputs, updates, givens)
}

fun function(inputs: List<Any> = emptyList(),
             outputs: Output,
             updates: List<Any> = emptyList(),
             givens: List<Pair<Output, Any>> = emptyList()): Function {
  val _givens = List(givens.size) {
    val p = givens[it]
    p.first to NDArray.toNDArray<Any>(p.second)
  }
  return FunctionTensor(inputs, listOf(outputs), updates, _givens)
}

fun function(inputs: List<Any> = emptyList(),
             outputs: List<Output> = emptyList(),
             updates: List<Any> = emptyList(),
             givens: List<Pair<Output, Any>> = emptyList()): Function {
  val _givens = List(givens.size) {
    val p = givens[it]
    p.first to NDArray.toNDArray<Any>(p.second)
  }
  return FunctionTensor(inputs, outputs, updates, _givens)
}

fun huber_loss(x: Output, delta: Float = 1f): Output {
  return tf.where(
      tf.less({ tf.abs(x) }, { tf.const(delta, it) }),
      tf.square(x) * 0.5f,
      delta * (tf.abs(x) - 0.5f * delta)
  )
}
