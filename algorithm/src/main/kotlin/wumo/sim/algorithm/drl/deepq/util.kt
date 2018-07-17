package wumo.sim.algorithm.drl.deepq

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.bytedeco.javacpp.tensorflow.DT_INT32
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.core.Space
import wumo.sim.spaces.Box
import wumo.sim.spaces.Discrete
import wumo.sim.util.Dimension
import wumo.sim.util.dim
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.tuple2
import wumo.sim.util.x

fun function(inputs: Array<Tensor>? = null, outputs: Tensor? = null, givens: Any? = null, updates: List<Any>? = null): Function {
  TODO()
}

fun huber_loss(x: Tensor, delta: Float = 1f): Tensor {
  val delta = tf.const(delta)
  return tf.where(
      tf.less(tf.abs(x), delta),
      tf.square(x) * tf.const(0.5f),
      delta * (tf.abs(x) - tf.const(0.5f) * delta)
  )
}

abstract class TfInput(val name: String = "(unnamed)") {
  abstract fun get(): Tensor
  abstract fun make_feed_dict(data: NDArray<*>): Pair<Tensor, NDArray<*>>
}

open class PlaceholderTfInput(val placeholder: Tensor) : TfInput(placeholder.name) {
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
        val input_x = tf.placeholder(shape = dim(batch_size), dtype = DT_INT32, name = name)
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

class ObservationInput(input: Tensor, val processed_input: Tensor) : PlaceholderTfInput(input) {
  companion object {
    operator fun invoke(observation_space: Space<*>, name: String = ""): ObservationInput {
      val (input, processed_input) = observation_input(observation_space, name = name)
      return ObservationInput(input, processed_input)
    }
  }
  
  override fun get() = processed_input
}