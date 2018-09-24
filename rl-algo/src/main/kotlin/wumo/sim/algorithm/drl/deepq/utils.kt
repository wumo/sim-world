package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.drl.common.observation_input
import wumo.sim.core.Space
import wumo.sim.tensorflow.ops.Output
import wumo.sim.util.ndarray.NDArray
abstract class TfInput(val name: String = "(unnamed)") {
  abstract fun get(): Output
  abstract fun make_feed_dict(data: NDArray<*>): Pair<Output, NDArray<*>>
}

open class PlaceholderTfInput(val placeholder: Output) : TfInput(placeholder.name) {
  override fun get() = placeholder
  
  override fun make_feed_dict(data: NDArray<*>) = placeholder to data
}

class ObservationInput(input: Output, val processed_input: Output) : PlaceholderTfInput(input) {
  companion object {
    operator fun invoke(observation_space: Space<*,*>, name: String = ""): ObservationInput {
      val (input, processed_input) = observation_input(observation_space, name = name)
      return ObservationInput(input, processed_input)
    }
  }
  
  override fun get() = processed_input
}