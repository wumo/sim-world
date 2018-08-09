package wumo.sim.tensorflow.contrib

import org.bytedeco.javacpp.tensorflow.DT_INT32
import org.bytedeco.javacpp.tensorflow.DT_INT64
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.base_dtype
import wumo.sim.tensorflow.layers.Dense
import wumo.sim.tensorflow.layers.TensorFunction
import wumo.sim.tensorflow.ops.variables.xavier_initializer
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.gen.oneHot
import wumo.sim.tensorflow.ops.gen.relu
import wumo.sim.tensorflow.tf

@Suppress("NAME_SHADOWING")
fun TF.one_hot_encoding(labels: Output,
                        num_class: Int,
                        on_value: Float = 1f, off_value: Float = 0f,
                        name: String = "OneHotEncoding"): Output {
  name_scope(name) {
    val labels = if (labels.dtype == DT_INT32) cast(labels, DT_INT64) else labels
    return oneHot(labels, const(num_class, name = "depth"),
                  const(on_value, name = "on_value"),
                  const(off_value, name = "off_value"), name = ctxNs.scopeName)
  }
}

fun TF.fully_connected(inputs: Output,
                       num_outputs: Int,
                       activation_fn: TensorFunction? = { tf.relu(it) },
                       normalizer_fn: ((Output, Any?) -> Output)? = null,
                       normalizer_params: Any? = null,
                       weights_initializer: Initializer = xavier_initializer(),
                       weights_regularizer: TensorFunction? = null,
                       biases_initializer: Initializer? = zeros_initializer(),
                       biases_regularizer: TensorFunction? = null,
                       variables_collections: Any? = null,
                       outputs_collections: Any? = null,
                       trainable: Boolean = true): Output {
  variable_scope("fully_connected") {
    val layer = Dense(units = num_outputs,
                      activation = null,
                      use_bias = normalizer_fn == null && biases_initializer != null,
                      kernel_initializer = weights_initializer,
                      bias_initializer = biases_initializer,
                      kernel_regularizer = weights_regularizer,
                      bias_regularizer = biases_regularizer,
                      activity_regularizer = null,
                      trainable = trainable,
                      dtype = inputs.dtype)
    var outputs = layer(inputs)
    
    if (normalizer_fn != null)
      outputs = normalizer_fn(outputs, normalizer_params)
    
    if (activation_fn != null)
      outputs = activation_fn(outputs)
    return outputs
  }
}

/**
 * Adds a Layer Normalization layer.
 * Based on the paper:

"Layer Normalization"

Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

https://arxiv.org/abs/1607.06450.

Can be used as a normalizer function for conv2d and fully_connected.

Given a tensor `inputs` of rank `R`, moments are calculated and normalization
is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
if requested, is performed over axes `begin_params_axis .. R - 1`.

By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
meaning that normalization is performed over all but the first axis
(the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
parameters are calculated for the rightmost axis (the `C` if `inputs` is
`NHWC`).  Scaling and recentering is performed via broadcast of the
`beta` and `gamma` parameters with the normalized tensor.

The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
and this part of the inputs' shape must be fully defined.

Args:
 * @param inputs: A tensor having rank `R`. The normalization is performed over
axes `begin_norm_axis ... R - 1` and centering and scaling parameters
are calculated over `begin_params_axis ... R - 1`.
 * @param center: If True, add offset of `beta` to normalized tensor. If False, `beta`
is ignored.
 * @param scale: If True, multiply by `gamma`. If False, `gamma` is
not used. When the next layer is linear (also e.g. `nn.relu`), this can be
disabled since the scaling can be done by the next layer.
 * @param activation_fn: Activation function, default set to None to skip it and
maintain a linear activation.
 * @param trainable: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
 * @param begin_norm_axis: The first normalization dimension: normalization will be
performed along dimensions `begin_norm_axis : rank(inputs)`
 * @param begin_params_axis: The first parameter (beta, gamma) dimension: scale
and centering parameters will have dimensions
`begin_params_axis : rank(inputs)` and will be broadcast with the
normalized inputs accordingly.
 
 * @return:A `Output` representing the output of the findOp, having the same
shape and dtype as `inputs`.
 
 * @throws
Raises:
ValueError: If the rank of `inputs` is not known at graph build time,
or if `inputs.shape[begin_params_axis:]` is not fully defined at
graph build time.
 */
fun TF.layer_norm(inputs: Output,
                  center: Boolean = true, scale: Boolean = true,
                  activation_fn: TensorFunction? = null,
                  trainable: Boolean = true,
                  _begin_norm_axis: Int = 1,
                  begin_params_axis: Int = -1): Output {
  variable_scope("LayerNorm") {
    val inputs_shape = inputs.shape
    val inputs_rank = inputs_shape.rank()
    val dtype = inputs.dtype.base_dtype
    val begin_norm_axis = if (_begin_norm_axis < 0) {
      inputs_rank + _begin_norm_axis
    } else _begin_norm_axis
    if (begin_params_axis >= inputs_rank || begin_norm_axis >= inputs_rank)
      throw IllegalArgumentException("begin_params_axis ($begin_params_axis) and begin_norm_axis ($begin_norm_axis) " +
                                         "must be < rank(inputs) ($inputs_rank)")
    val params_shape = inputs_shape[-1 until 0]
    if (!params_shape.is_fully_defined)
      throw IllegalArgumentException("Inputs ${inputs.name}: shape(inputs)[$begin_params_axis:]" +
                                         " is not fully defined: $inputs_shape")
    //Allocate parameters for the beta and gamma of the normalization.
    val beta = if (center)
      tf.get_variable(params_shape, dtype, zeros_initializer(), "beta", trainable)
    else null
    val gamma = if (scale)
      tf.get_variable(params_shape, dtype, ones_initializer(), "gamma", trainable)
    else null
    //Calculate the moments on the last axis (layer activations).
    val norm_axes = (begin_norm_axis until inputs_rank).map { it.toLong() }.toLongArray()
    val (mean, variance) = tf.moments(inputs, norm_axes, keep_dims = true)
    //Compute layer normalization using the batch_normalization function.
    var outputs = tf.batch_normalization(
        inputs,
        mean,
        variance,
        offset = beta,
        scale = gamma,
        variance_epsilon = 1e-12f)
    outputs.set_shape(inputs_shape)
    if (activation_fn != null)
      outputs = activation_fn(outputs)
    return outputs
  }
}