package wumo.sim.algorithm.drl.common

import wumo.sim.core.Space
import wumo.sim.spaces.Box
import wumo.sim.spaces.Discrete
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.toDataType
import wumo.sim.util.NONE
import wumo.sim.util.Shape
import wumo.sim.util.t2

fun observationPlaceholder(obSpace: Space<*>,
                           batch_size: Int = -1,
                           name: String = "Ob"): Output {
  assert(obSpace is Discrete || obSpace is Box) {
    "Can only deal with Discrete and Box observation spaces for now"
  }
  
  return tf.placeholder(
      Shape(intArrayOf(batch_size, *obSpace.shape.asIntArray()!!)),
      obSpace.dataType.toDataType(),
      name)
}

/**
 * Build observation input with encoding depending on the
 * observation space type
 *
 */
fun observation_input(obSpace: Space<*>,
                      batchSize: Int = -1,
                      name: String = "Ob"): t2<Output, Output> {
  val placeholder = observationPlaceholder(obSpace, batchSize, name)
  return t2(placeholder, encodeObservation(obSpace, placeholder))
}

fun encodeObservation(obSpace: Space<*>, placeholder: Output): Output =
    when (obSpace) {
      is Discrete -> tf.toFloat(tf.oneHot(placeholder, tf.const(obSpace.n)))
      is Box -> tf.toFloat(placeholder)
      else -> NONE()
    }