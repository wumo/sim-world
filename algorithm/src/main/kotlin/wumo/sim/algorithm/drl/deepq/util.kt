@file:Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")

package wumo.sim.algorithm.drl.deepq

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.bytedeco.javacpp.tensorflow.DT_INT32
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.ops.gen.abs
import wumo.sim.algorithm.tensorflow.ops.gen.less
import wumo.sim.algorithm.tensorflow.ops.gen.oneHot
import wumo.sim.algorithm.tensorflow.ops.gen.square
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.core.Space
import wumo.sim.spaces.Box
import wumo.sim.spaces.Discrete
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.tuple2
import wumo.sim.util.x

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
    return tf.session.run(outputs, updates, feed_dict = feed_dict)
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
    return tf.session.run(outputs, updates, feed_dict = feed_dict)
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

abstract class TfInput(val name: String = "(unnamed)") {
  abstract fun get(): Output
  abstract fun make_feed_dict(data: NDArray<*>): Pair<Output, NDArray<*>>
}

open class PlaceholderTfInput(val placeholder: Output) : TfInput(placeholder.name) {
  override fun get() = placeholder
  
  override fun make_feed_dict(data: NDArray<*>) = placeholder to data
}

/**
 * Build observation input with encoding depending on the
 * observation space type
 *
 */
fun observation_input(ob_space: Space<*>, batch_size: Int = -1, name: String = "Ob") =
    when (ob_space) {
      is Discrete -> {
        val input_x = tf.placeholder(shape = Shape(batch_size), dtype = DT_INT32, name = name)
        val processed_x = tf.cast(tf.oneHot(input_x, tf.const(ob_space.n), tf.const(0), tf.const(1)), DT_FLOAT)
        tuple2(input_x, processed_x)
      }
      is Box -> {
        val input_shape = batch_size x ob_space.shape
        val input_x = tf.placeholder(shape = input_shape, dtype = ob_space.dtype, name = name)
        val processed_x = tf.cast(input_x, DT_FLOAT)
        tuple2(input_x, processed_x)
      }
      else -> throw NotImplementedError()
    }

class ObservationInput(input: Output, val processed_input: Output) : PlaceholderTfInput(input) {
  companion object {
    operator fun invoke(observation_space: Space<*>, name: String = ""): ObservationInput {
      val (input, processed_input) = observation_input(observation_space, name = name)
      return ObservationInput(input, processed_input)
    }
  }
  
  override fun get() = processed_input
}