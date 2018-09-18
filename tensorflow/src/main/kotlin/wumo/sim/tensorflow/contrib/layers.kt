package wumo.sim.tensorflow.contrib

import wumo.sim.tensorflow.contrib.layers.ConvPadding.SAME
import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.layers.convolutional.Conv1D
import wumo.sim.tensorflow.layers.convolutional.Conv2D
import wumo.sim.tensorflow.layers.core.Dense
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_first
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_last
import wumo.sim.tensorflow.ops.DeviceFunction
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.*
import wumo.sim.tensorflow.ops.variables.Variable.VariableGetter
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.tf.variableScope
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.Shape
import wumo.sim.util.errorIf
import wumo.sim.tensorflow.layers.core.layers as core_layers

typealias NormalizerFunction = (Output, Any?) -> Output

object layers {
  enum class ConvPadding {
    SAME, VALID
  }
  
  enum class CNNDataFormat {
    NWC, NCW, NHWC, NCHW, NDHWC, NCDHW
  }
  
  fun convolution(inputs: Output,
                  num_outputs: Int,
                  kernel_size: Int,
                  stride: Int = 1,
                  padding: ConvPadding = SAME,
                  data_format: CNNDataFormat? = null,
                  dilation_rate: Int = 1,
                  activation_fn: TensorFunction? = { tf.relu(it) },
                  normalizer_fn: NormalizerFunction? = null,
                  normalizer_params: Map<String, Any>? = null,
                  weights_initializer: Initializer = tf.xavier_initializer(),
                  weights_regularizer: TensorFunction? = null,
                  biases_initializer: Initializer? = tf.zerosInitializer(),
                  biases_regularizer: TensorFunction? = null,
                  reuse: Reuse = ReuseOrCreateNew,
                  trainable: Boolean = true,
                  scope: String = "Conv",
                  conv_dims: Int? = null): Output =
      variableScope(scope, reuse = reuse, isDefaultName = true) {
        val input_rank = inputs.shape.rank
        errorIf(conv_dims != null && conv_dims + 2 != input_rank) {
          "Convolution expects input with rank ${conv_dims!! + 2}, got $input_rank"
        }
        val layerClass = when (input_rank) {
          3 -> ::Conv1D
          4 -> ::Conv2D
//        5 -> TODO()
          else -> error("Convolution not supported for input with rank $input_rank")
        }
        val df = if (data_format != null && data_format.name.startsWith("NC"))
          channels_first
        else channels_last
        val layer = layerClass(
            num_outputs,
            kernel_size,
            stride,
            padding,
            df,
            dilation_rate,
            null,
            normalizer_fn == null && biases_initializer != null,
            weights_initializer,
            biases_initializer,
            weights_regularizer,
            biases_regularizer,
            null,
            null,
            null,
            trainable,
            inputs.dataType.baseDataType,
            tf.currentVariableScope.name,
            tf.currentVariableScope,
            reuse)
        var outputs = layer(inputs)
        if (normalizer_fn != null)
          outputs = normalizer_fn(outputs, normalizer_params)
        if (activation_fn != null)
          outputs = activation_fn(outputs)!!
        outputs
      }
  
  fun convolution1d(inputs: Output,
                    num_outputs: Int,
                    kernel_size: Int,
                    stride: Int = 1,
                    padding: ConvPadding = SAME,
                    data_format: CNNDataFormat? = null,
                    rate: Int = 1,
                    activation_fn: TensorFunction? = { tf.relu(it) },
                    normalizer_fn: NormalizerFunction? = null,
                    normalizer_params: Map<String, Any>? = null,
                    weights_initializer: Initializer = tf.xavier_initializer(),
                    weights_regularizer: TensorFunction? = null,
                    biases_initializer: Initializer? = tf.zerosInitializer(),
                    biases_regularizer: TensorFunction? = null,
                    reuse: Reuse = ReuseOrCreateNew,
                    trainable: Boolean = true,
                    scope: String): Output =
      convolution(inputs, num_outputs, kernel_size, stride,
                  padding, data_format,
                  rate, activation_fn,
                  normalizer_fn, normalizer_params,
                  weights_initializer, weights_regularizer,
                  biases_initializer, biases_regularizer,
                  reuse, trainable, scope, 1)
  
  fun convolution2d(inputs: Output,
                    num_outputs: Int,
                    kernel_size: Int,
                    stride: Int = 1,
                    padding: ConvPadding = SAME,
                    data_format: CNNDataFormat? = null,
                    rate: Int = 1,
                    activation_fn: TensorFunction? = { tf.relu(it) },
                    normalizer_fn: NormalizerFunction? = null,
                    normalizer_params: Map<String, Any>? = null,
                    weights_initializer: Initializer = tf.xavier_initializer(),
                    weights_regularizer: TensorFunction? = null,
                    biases_initializer: Initializer? = tf.zerosInitializer(),
                    biases_regularizer: TensorFunction? = null,
                    reuse: Reuse = ReuseOrCreateNew,
                    trainable: Boolean = true,
                    scope: String): Output =
      convolution(inputs, num_outputs, kernel_size, stride,
                  padding, data_format,
                  rate, activation_fn,
                  normalizer_fn, normalizer_params,
                  weights_initializer, weights_regularizer,
                  biases_initializer, biases_regularizer,
                  reuse, trainable, scope, 2)
  
  fun convolution3d(inputs: Output,
                    num_outputs: Int,
                    kernel_size: Int,
                    stride: Int = 1,
                    padding: ConvPadding = SAME,
                    data_format: CNNDataFormat? = null,
                    rate: Int = 1,
                    activation_fn: TensorFunction? = { tf.relu(it) },
                    normalizer_fn: NormalizerFunction? = null,
                    normalizer_params: Map<String, Any>? = null,
                    weights_initializer: Initializer = tf.xavier_initializer(),
                    weights_regularizer: TensorFunction? = null,
                    biases_initializer: Initializer? = tf.zerosInitializer(),
                    biases_regularizer: TensorFunction? = null,
                    reuse: Reuse = ReuseOrCreateNew,
                    trainable: Boolean = true,
                    scope: String): Output =
      convolution(inputs, num_outputs, kernel_size, stride,
                  padding, data_format,
                  rate, activation_fn,
                  normalizer_fn, normalizer_params,
                  weights_initializer, weights_regularizer,
                  biases_initializer, biases_regularizer,
                  reuse, trainable, scope, 3)
  
  @Suppress("NAME_SHADOWING")
  fun one_hot_encoding(labels: Output,
                       num_class: Int,
                       on_value: Float = 1f, off_value: Float = 0f,
                       name: String = "OneHotEncoding"): Output =
      tf.nameScope(name) {
        val labels = if (labels.dataType == INT32) tf.toInt64(labels) else labels
        tf.oneHot(labels, tf.const(num_class, name = "depth"),
                  tf.const(on_value, name = "on_value"),
                  tf.const(off_value, name = "off_value"), name = tf.currentNameScope)
      }
  
  fun flatten(inputs: Output, scope: String = "Flatten"): Output =
      tf.nameScope(scope) {
        core_layers.flatten(inputs)
      }
  
  private fun buildVariableGetter(rename: Map<String, String>): VariableGetter =
      object : VariableGetter {
        override fun invoke(name: String,
                            dataType: DataType<*>?,
                            shape: Shape?,
                            initializer: Initializer?,
                            regularizer: Regularizer?,
                            trainable: Boolean,
                            reuse: Reuse,
                            collections: MutableSet<Graph.Graph.Key<Variable>>,
                            cachingDevice: DeviceFunction?,
                            underlyingGetter: VariableGetter?): Variable {
          val nameParts = name.split('/').toMutableList()
          val shortName = nameParts.last()
          val name = if (shortName in rename) {
            nameParts[nameParts.lastIndex] = rename[shortName]!!
            nameParts.joinToString("/")
          } else name
          return tf.modelVariable(name, shape, dataType, initializer, regularizer,
                                  trainable, reuse, collections, cachingDevice)
        }
      }
  
  fun fullyConnected(inputs: Output,
                     num_outputs: Int,
                     activation_fn: TensorFunction? = { tf.relu(it) },
                     normalizer_fn: NormalizerFunction? = null,
                     normalizer_params: Map<String, Any>? = null,
                     weights_initializer: Initializer = tf.xavier_initializer(),
                     weights_regularizer: TensorFunction? = null,
                     biases_initializer: Initializer? = tf.zerosInitializer(),
                     biases_regularizer: TensorFunction? = null,
                     reuse: Reuse = ReuseOrCreateNew,
                     trainable: Boolean = true,
                     scope: String = "fullyConnected"): Output =
      variableScope(scope, reuse, isDefaultName = true) {
        val layer = Dense(units = num_outputs,
                          activation = null,
                          use_bias = normalizer_fn == null && biases_initializer != null,
                          kernel_initializer = weights_initializer,
                          bias_initializer = biases_initializer,
                          kernel_regularizer = weights_regularizer,
                          bias_regularizer = biases_regularizer,
                          activity_regularizer = null,
                          trainable = trainable,
                          name = tf.currentVariableScope.name,
                          dataType = inputs.dataType.baseDataType,
                          _scope = tf.currentVariableScope)
        var outputs = layer(inputs)
        //Apply normalizer function / layer.
        if (normalizer_fn != null)
          outputs = normalizer_fn(outputs, normalizer_params)
        if (activation_fn != null)
          outputs = activation_fn(outputs)!!
        outputs
      }
  
  fun layerNorm(inputs: Output, center: Boolean, scale: Boolean,
                activation_fn: TensorFunction? = null,
                reuse: Reuse = ReuseOrCreateNew,
                trainable: Boolean = true,
                begin_norm_axis: Int = 1,
                begin_params_axis: Int = 1): Output =
      variableScope("LayerNorm", reuse = reuse, isDefaultName = true) {
        val inputs_shape = inputs.shape
        val inputs_rank = inputs_shape.rank
        require(inputs_rank >= 0) { "Inputs ${inputs.name} has undefined rank." }
        val dtype = inputs.dataType.baseDataType
        val _begin_norm_axis = if (begin_norm_axis < 0) inputs_rank + begin_norm_axis
        else begin_norm_axis
        errorIf(begin_params_axis >= inputs_rank || _begin_norm_axis >= inputs_rank) {
          "begin_params_axis ($begin_params_axis) and begin_norm_axis " +
              "($_begin_norm_axis) must be < rank(inputs) ($inputs_rank)"
        }
        val params_shape = inputs_shape.slice(begin_params_axis)
        errorIf(!params_shape.isFullyDefined) {
          "Inputs ${inputs.name}: shape(inputs)[$begin_params_axis:] is not fully defined:" +
              " $inputs_shape"
        }
        val beta = if (center)
          tf.modelVariable("beta",
                           shape = params_shape,
                           dataType = dtype,
                           initializer = tf.zerosInitializer(),
                           trainable = trainable)
        else null
        val gamma = if (scale)
          tf.modelVariable("gamma",
                           shape = params_shape,
                           dataType = dtype,
                           initializer = tf.onesInitializer(),
                           trainable = trainable)
        else null
        val norm_axes = (_begin_norm_axis until inputs_rank).map { it.toLong() }.toLongArray()
        val (mean, variance) = tf.moments(inputs, norm_axes, keep_dims = true)
        //Compute layer normalization using the batchNormalization function.
        var outputs = tf.batchNormalization(
            inputs,
            mean,
            variance,
            offset = beta?.toOutput(),
            scale = gamma?.toOutput(),
            variance_epsilon = 1e-12f)
        outputs.setShape(inputs_shape)
        if (activation_fn != null)
          outputs = activation_fn(outputs)!!
        outputs
      }
}

//fun TF.layerNorm(inputs: Output,
//                  center: Boolean = true, scale: Boolean = true,
//                  activation_fn: TensorFunction? = null,
//                  trainable: Boolean = true,
//                  _begin_norm_axis: Int = 1,
//                  begin_params_axis: Int = -1): Output {
//  variable_scope("LayerNorm") {
//    val inputs_shape = inputs.shape
//    val inputs_rank = inputs_shape.rank()
//    val dataType = inputs.dataType.base_dtype
//    val begin_norm_axis = if (_begin_norm_axis < 0) {
//      inputs_rank + _begin_norm_axis
//    } else _begin_norm_axis
//    if (begin_params_axis >= inputs_rank || begin_norm_axis >= inputs_rank)
//      throw IllegalArgumentException("begin_params_axis ($begin_params_axis) and begin_norm_axis ($begin_norm_axis) " +
//                                         "must be < rank(inputs) ($inputs_rank)")
//    val params_shape = inputs_shape[-1 until 0]
//    if (!params_shape.isFullyDefined)
//      throw IllegalArgumentException("Inputs ${inputs.name}: shape(inputs)[$begin_params_axis:]" +
//                                         " is not fully defined: $inputs_shape")
//    //Allocate parameters for the beta and gamma of the normalization.
//    val beta = if (center)
//      tf.get_variable(params_shape, dataType, zerosInitializer(), "beta", trainable)
//    else null
//    val gamma = if (scale)
//      tf.get_variable(params_shape, dataType, onesInitializer(), "gamma", trainable)
//    else null
//    //Calculate the moments on the last axis (layer activations).
//    val norm_axes = (begin_norm_axis until inputs_rank).map { it.toLong() }.toLongArray()
//    val (mean, variance) = tf.moments(inputs, norm_axes, keep_dims = true)
//    //Compute layer normalization using the batchNormalization function.
//    var outputs = tf.batchNormalization(
//        inputs,
//        mean,
//        variance,
//        offset = beta,
//        scale = gamma,
//        variance_epsilon = 1e-12f)
//    outputs.setShape(inputs_shape)
//    if (activation_fn != null)
//      outputs = activation_fn(outputs)
//    return outputs
//  }
//}