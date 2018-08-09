package wumo.sim.tensorflow.layers

import org.bytedeco.javacpp.tensorflow.DT_INVALID
import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.algorithm.tensorflow.ops.Initializer
import wumo.sim.algorithm.tensorflow.ops.get_variable
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Shape

typealias TensorFunction = (Output) -> Output

open class Layer(val trainable: Boolean = true,
                 val activity_reqularizer: Any? = null,
                 var dtype: Int = 0) {
  var statefule = false
  var built = false
  val losses = mutableListOf<Output>()
  
  open fun build(input_shape: Shape) {
    built = true
  }
  
  open fun call(input: Output) = input
  
  open operator fun invoke(inputs: Output): Output {
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
  
  protected fun add_variable(shape: Shape, dtype: Int,
                             initializer: Initializer,
                             regularizer: TensorFunction? = null,
                             trainable: Boolean = true,
                             name: String = ""): Output {
    val v = tf.get_variable(shape, dtype, initializer, name, trainable && this.trainable)
    if (regularizer != null)
      losses += regularizer(v)
    return v
  }
}