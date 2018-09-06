package wumo.sim.algorithm.drl.common

import wumo.sim.algorithm.drl.deepq.TfInput
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.minus
import wumo.sim.tensorflow.ops.basic.times
import wumo.sim.tensorflow.tf
import wumo.sim.util.a
import wumo.sim.util.ndarray.NDArray

interface Function {
  operator fun invoke(vararg args: Any): Array<NDArray<*>>
}

class FunctionTensor(val inputs: Array<out Any>,
                     val outputs: Array<Output>,
                     val updates: Array<out Any>,
                     val givens: Array<out Pair<Output, NDArray<*>>>) : Function {
  
  override operator fun invoke(vararg args: Any): Array<NDArray<*>> {
    assert(args.size <= inputs.size) { "Too many arguments provided" }
    val feed_dict = mutableMapOf<Output, NDArray<*>>()
    for (i in 0 until inputs.size) {
      val input = inputs[i]
      val value = NDArray.toNDArray(args[i])
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

class FunctionString(val inputs: Array<String>,
                     val outputs: Array<String>,
                     val updates: Array<String>,
                     val givens: Array<out Pair<String, NDArray<*>>>) : Function {
  
  override operator fun invoke(vararg args: Any): Array<NDArray<*>> {
    assert(args.size <= inputs.size) { "Too many arguments provided" }
    val feed_dict = mutableMapOf<String, NDArray<*>>()
    for (i in 0 until inputs.size) {
      val input = inputs[i]
      val value = NDArray.toNDArray(args[i])
      feed_dict += input to value
    }
    for ((input, value) in givens)
      feed_dict.putIfAbsent(input, value)
    return tf.currentSession!!.run(outputs, updates, feed_dict = feed_dict)
  }
}

fun function(inputs: Array<String> = emptyArray(),
             outputs: Array<String> = emptyArray(),
             updates: Array<String> = emptyArray(),
             givens: Array<Pair<String, NDArray<*>>> = emptyArray()): Function {
  return FunctionString(inputs, outputs, updates, givens)
}

fun function(inputs: Array<out Any> = emptyArray(),
             outputs: Output,
             updates: Array<out Any> = emptyArray(),
             givens: Array<Pair<Output, Any>> = emptyArray()): Function {
  val _givens = a(givens.size) {
    val p = givens[it]
    p.first to NDArray.toNDArray(p.second)
  }
  return FunctionTensor(inputs, a(outputs), updates, _givens)
}

fun function(inputs: Array<out Any> = emptyArray(),
             outputs: Array<Output> = emptyArray(),
             updates: Array<out Any> = emptyArray(),
             givens: Array<Pair<Output, Any>> = emptyArray()): Function {
  val _givens = a(givens.size) {
    val p = givens[it]
    p.first to NDArray.toNDArray(p.second)
  }
  return FunctionTensor(inputs, outputs, updates, _givens)
}

fun huber_loss(x: Output, delta: Float = 1f): Output {
  val delta = tf.const(delta)
  return tf.where(
      tf.less(tf.abs(x), delta),
      tf.square(x) * tf.const(0.5f),
      delta * (tf.abs(x) - tf.const(0.5f) * delta)
  )
}
