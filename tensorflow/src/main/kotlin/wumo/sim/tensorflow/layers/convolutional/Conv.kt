package wumo.sim.tensorflow.layers.convolutional

import wumo.sim.tensorflow.contrib.layers
import wumo.sim.tensorflow.contrib.layers.ConvPadding
import wumo.sim.tensorflow.contrib.layers.ConvPadding.VALID
import wumo.sim.tensorflow.contrib.layers.CNNDataFormat.*
import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.layers.Layer
import wumo.sim.tensorflow.layers.utils.DataFormat
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_first
import wumo.sim.tensorflow.layers.utils.DataFormat.channels_last
import wumo.sim.tensorflow.layers.utils.toCNNDataFormat
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.basic.nn_ops
import wumo.sim.tensorflow.ops.basic.plus
import wumo.sim.tensorflow.ops.basic.toOutput
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Reuse
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.ops.variables.VariableScope
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape
import wumo.sim.util.errorIf

open class Conv(val rank: Int,
                val filters: Int,
                kernel_size: Int,
                strides: Int = 1,
                val padding: ConvPadding = VALID,
                val data_format: DataFormat = channels_last,
                dilation_rate: Int = 1,
                val activation: TensorFunction? = null,
                val use_bias: Boolean = true,
                val kernel_initializer: Initializer = tf.glorotUniformInitializer(),
                val bias_initializer: Initializer? = tf.zerosInitializer(),
                val kernel_regularizer: TensorFunction? = null,
                val bias_regularizer: TensorFunction? = null,
                activity_regularizer: Any? = null,
                val kernel_constraint: Any? = null,
                val bias_constraint: Any? = null,
                trainable: Boolean = true,
                dataType: DataType<*>,
                name: String,
                _scope: VariableScope,
                _reuse: Reuse) : Layer(name,
                                       trainable,
                                       dataType = dataType,
                                       activity_reqularizer = activity_regularizer,
                                       _reuse=_reuse,
                                       _scope = _scope) {
  
  val kernel_size = List(rank) { kernel_size }
  val strides = List(rank) { strides }
  val dilation_rate = List(rank) { dilation_rate }
  
  lateinit var kernel: Variable
  var bias: Variable? = null
  lateinit var convolution_op: nn_ops.Convolution
  
  override fun build(input_shape: Shape) {
    val channel_axis = if (data_format == channels_first) 1 else -1
    errorIf(input_shape[channel_axis] == -1) {
      "The channel dimension of the inputs should be defined. Found `None`."
    }
    val input_dim = input_shape[channel_axis]
    val kernel_shape = Shape(intArrayOf(*kernel_size.toIntArray(), input_dim, filters))
    
    kernel = addWeight(name = "weights",
                       shape = kernel_shape,
                       initializer = kernel_initializer,
                       regularizer = kernel_regularizer,
                       constraint = kernel_constraint,
                       trainable = true,
                       dataType = dataType)
    if (use_bias)
      bias = addWeight(name = "biases",
                       shape = Shape(filters),
                       initializer = bias_initializer!!,
                       regularizer = bias_regularizer,
                       constraint = bias_constraint,
                       trainable = true,
                       dataType = dataType)
    convolution_op = nn_ops.Convolution(
        input_shape,
        fileterShape = kernel.shape,
        dilation_rate = dilation_rate,
        strides = strides,
        padding = padding,
        data_format = data_format.toCNNDataFormat(rank + 2))
    built = true
  }
  
  override fun call(input: Output): Output {
    var outputs = convolution_op(input, kernel.toOutput())
    if (use_bias) {
      val bias = bias!!
      when (data_format) {
        channels_first -> {
          when (rank) {
            1 -> {
              val bias = tf.reshape(bias.toOutput(), Shape(1, filters, 1).toOutput())
              outputs += bias
            }
            2 ->
              outputs = tf.biasAdd(outputs, bias.toOutput(), dataFormat = NCHW.name)
            3 -> {
              val outputs_shape = outputs.shape
              var outputs_4d = tf.reshape(outputs,
                                          Shape(outputs_shape[0], outputs_shape[1],
                                                outputs_shape[2] * outputs_shape[3],
                                                outputs_shape[4]).toOutput())
              outputs_4d = tf.biasAdd(outputs_4d, bias.toOutput(), dataFormat = NCHW.name)
              outputs = tf.reshape(outputs_4d, outputs_shape.toOutput())
            }
          }
        }
        channels_last -> {
          outputs = tf.biasAdd(outputs, bias.toOutput(), dataFormat = NHWC.name)
        }
      }
    }
    if (activation != null)
      outputs = activation!!(outputs)!!
    return outputs
  }
}