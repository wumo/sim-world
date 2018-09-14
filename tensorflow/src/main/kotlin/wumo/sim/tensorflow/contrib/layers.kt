package wumo.sim.tensorflow.contrib

import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.layers.core.Dense
import wumo.sim.tensorflow.ops.DeviceFunction
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.variables.*
import wumo.sim.tensorflow.ops.variables.Variable.VariableGetter
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.Shape
import wumo.sim.util.errorIf
import wumo.sim.tensorflow.layers.core.layers as core_layers

object layers {
  
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
  
  fun fully_connected(inputs: Output,
                      num_outputs: Int,
                      activation_fn: TensorFunction? = { tf.relu(it) },
                      normalizer_fn: ((Output, Any?) -> Output)? = null,
                      normalizer_params: Map<String, Any>? = null,
                      weights_initializer: Initializer = tf.xavier_initializer(),
                      weights_regularizer: TensorFunction? = null,
                      biases_initializer: Initializer? = tf.zerosInitializer(),
                      biases_regularizer: TensorFunction? = null,
                      reuse: Reuse = ReuseOrCreateNew,
                      trainable: Boolean = true,
                      scope: String = "fully_connected"): Output {
    return tf.variableScope(scope, reuse, isDefaultName = true) {
      val layer = Dense(units = num_outputs,
                        activation = null,
                        use_bias = normalizer_fn == null && biases_initializer != null,
                        kernel_initializer = weights_initializer,
                        bias_initializer = biases_initializer,
                        kernel_regularizer = weights_regularizer,
                        bias_regularizer = biases_regularizer,
                        activity_regularizer = null,
                        trainable = trainable,
                        name = VariableScope.current.name,
                        dataType = inputs.dataType.baseDataType,
                        _scope = VariableScope.current)
      var outputs = layer(inputs)
      //Apply normalizer function / layer.
      if (normalizer_fn != null)
        outputs = normalizer_fn(outputs, normalizer_params ?: mapOf<String, Any>())
      if (activation_fn != null)
        outputs = activation_fn(outputs)!!
      outputs
    }
  }
  
  fun layer_norm(inputs: Output, center: Boolean, scale: Boolean,
                 activation_fn: TensorFunction? = null,
                 reuse: Reuse = ReuseOrCreateNew,
                 trainable: Boolean = true,
                 begin_norm_axis: Int = 1,
                 begin_params_axis: Int = 1): Output {
    tf.variableScope("LayerNorm", reuse = reuse, isDefaultName = true) {
      val inputs_shape = inputs.shape
      val inputs_rank = inputs_shape.rank
      require(inputs_rank >= 0) { "Inputs ${inputs.name} has undefined rank." }
      val dtype = inputs.dataType.baseDataType
      val begin_norm_axis = if (begin_norm_axis < 0) inputs_rank + begin_norm_axis
      else begin_norm_axis
      errorIf(begin_params_axis >= inputs_rank || begin_norm_axis >= inputs_rank) {
        "begin_params_axis ($begin_params_axis) and begin_norm_axis " +
            "($begin_norm_axis) must be < rank(inputs) ($inputs_rank)"
      }
      val params_shape = inputs_shape.slice(begin_params_axis)
      errorIf(!params_shape.isFullyDefined) {
        "Inputs ${inputs.name}: shape(inputs)[$begin_params_axis:] is not fully defined:" +
            " $inputs_shape"
      }
      var beta = if (center)
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
      
    }
  }
}

//fun TF.layer_norm(inputs: Output,
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
//    //Compute layer normalization using the batch_normalization function.
//    var outputs = tf.batch_normalization(
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