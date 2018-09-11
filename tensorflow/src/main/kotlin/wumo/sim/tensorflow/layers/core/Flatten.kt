package wumo.sim.tensorflow.layers.core

import wumo.sim.tensorflow.layers.Layer
import wumo.sim.tensorflow.layers.utils.DataFormat
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_first
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_last
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.get
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape

class Flatten(val dataFormat: DataFormat = channels_last,
              name: String = "flatten") : Layer(name = name) {
  
  override fun call(input: Output): Output {
    val input = if (dataFormat == channels_first) {
      input
    } else input
    val output = tf.reshape(input,
                            tf.stack(listOf(tf.shape(input)[0],
                                            -1)))
    output.setShape(computeOutputShape(input.shape))
    return output
  }
  
  fun computeOutputShape(inputShape: Shape): Shape {
    val d0 = inputShape[0]
    val slice = inputShape.slice(1)
    val d1 = if (slice.all { it >= 0 })
      slice.numElements()
    else
      -1
    return Shape(d0, d1)
  }
}