package wumo.sim.algorithm.tensorflow.contrib

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.layers.Dense
import wumo.sim.algorithm.tensorflow.layers.TensorFunction
import wumo.sim.algorithm.tensorflow.layers.xavier_initializer
import wumo.sim.algorithm.tensorflow.ops.*

@Suppress("NAME_SHADOWING")
fun TF.one_hot_encoding(labels: Tensor,
                        num_class: Int,
                        on_value: Float = 1f, off_value: Float = 0f,
                        name: String = "OneHotEncoding"): Tensor {
  subscope(name) {
    val labels = if (labels.dtype == DT_INT32) cast(labels, DT_INT64) else labels
    return oneHot(labels, const(num_class, name = "depth"),
                  const(on_value, name = "on_value"),
                  const(off_value, name = "off_value"), ctx.useContextName())
  }
}

fun TF.fully_connected(inputs: Tensor,
                       num_outputs: Int,
                       activation_fn: TensorFunction? = null,
                       normalizer_fn: ((Tensor, Any?) -> Tensor)? = null,
                       normalizer_params: Any? = null,
                       weights_initializer: Initializer = xavier_initializer(),
                       weights_tensorFunction: TensorFunction? = null,
                       biases_initializer: Initializer? = zeros_initializer(),
                       biases_tensorFunction: TensorFunction? = null,
                       reuse: Any? = null,
                       variables_collections: Any? = null,
                       outputs_collections: Any? = null,
                       trainable: Boolean = true): Tensor {
  val layer = Dense(this, units = num_outputs,
                    activation = null,
                    use_bias = normalizer_fn == null && biases_initializer != null,
                    kernel_initializer = weights_initializer,
                    bias_initializer = biases_initializer,
                    bias_tensorFunction = biases_tensorFunction,
                    kernel_tensorFunction = weights_tensorFunction,
                    activity_regularizer = null,
                    trainable = trainable,
                    name = "fully_connected",
                    dtype = inputs.dtype)
  var outputs = layer(inputs)
  
  if (normalizer_fn != null)
    outputs = normalizer_fn(outputs, normalizer_params)
  
  if (activation_fn != null)
    outputs = activation_fn(outputs)
  return outputs
}
