package wumo.sim.tensorflow.layers.core

import wumo.sim.tensorflow.ops.Output

object layers {
  fun flatten(inputs: Output): Output {
    val layer = Flatten()
    return layer(inputs)
  }
}