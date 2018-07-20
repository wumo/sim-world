package wumo.sim.algorithm.tensorflow.layers

import org.bytedeco.javacpp.tensorflow.DT_INVALID
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.Initializer
import wumo.sim.algorithm.tensorflow.ops.get_variable
import wumo.sim.algorithm.tensorflow.ops.variable
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Dimension

typealias TensorFunction = (Tensor) -> Tensor

open class Layer(val trainable: Boolean = true,
                 val activity_reqularizer: Any? = null,
                 var dtype: Int = 0) {
  var statefule = false
  var built = false
  val losses = mutableListOf<Tensor>()
  
  open fun build(input_shape: Dimension) {
    built = true
  }
  
  open fun call(input: Tensor) = input
  
  open operator fun invoke(inputs: Tensor): Tensor {
    if (!built) {
      if (dtype == DT_INVALID)
        dtype = inputs.dtype
      val input_shape = inputs.shape
      build(input_shape)
    }
    //if not in_deferred_mode:
    val outputs = call(inputs)
    if (activity_reqularizer != null) {
    }
    return outputs
  }
  
  protected fun add_variable(shape: Dimension, dtype: Int,
                             initializer: Initializer,
                             regularizer: TensorFunction? = null,
                             trainable: Boolean = true,
                             name: String = ""): Tensor {
    val v = tf.get_variable(shape, dtype, initializer, name, trainable && this.trainable)
    if (regularizer != null)
      losses += regularizer(v)
    return v
  }
}