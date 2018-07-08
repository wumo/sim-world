package wumo.sim.algorithm.tensorflow.layers

import org.bytedeco.javacpp.tensorflow.DT_INVALID
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.util.Dimension

open class Layer(val trainable: Boolean = true,
                 val activity_reqularizer: Any? = null,
                 var dtype: Int = 0,
                 val name: String = "") {
  var statefule = false
  var built = false
  
  open fun build(input_shapes: Dimension) {
    built = true
  }
  
  open fun call(inputs: Tensor) = inputs
  
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
}