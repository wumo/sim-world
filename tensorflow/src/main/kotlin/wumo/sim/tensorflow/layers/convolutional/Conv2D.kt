package wumo.sim.tensorflow.layers.convolutional

import wumo.sim.tensorflow.contrib.layers.ConvPadding
import wumo.sim.tensorflow.contrib.layers.ConvPadding.VALID
import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.layers.utils.DataFormat
import wumo.sim.tensorflow.ops.variables.Initializer
import wumo.sim.tensorflow.ops.variables.Reuse
import wumo.sim.tensorflow.ops.variables.VariableScope
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType

class Conv2D(filters: Int,
             kernel_size: Int,
             strides: Int = 1,
             padding: ConvPadding = VALID,
             data_format: DataFormat,
             dilation_rate: Int = 1,
             activation: TensorFunction? = null,
             use_bias: Boolean = true,
             kernel_initializer: Initializer = tf.glorotUniformInitializer(),
             bias_initializer: Initializer? = tf.zerosInitializer(),
             kernel_regularizer: TensorFunction? = null,
             bias_regularizer: TensorFunction? = null,
             activity_regularizer: Any? = null,
             kernel_constraint: Any? = null,
             bias_constraint: Any? = null,
             trainable: Boolean = true,
             dataType: DataType<*>,
             name: String,
             _scope: VariableScope,
             _reuse: Reuse) : Conv(2,
                                   filters, kernel_size, strides,
                                   padding, data_format, dilation_rate,
                                   activation, use_bias,
                                   kernel_initializer, bias_initializer,
                                   kernel_regularizer, bias_regularizer,
                                   activity_regularizer,
                                   kernel_constraint, bias_constraint,
                                   trainable,
                                   dataType,
                                   name,
                                   _scope,
                                   _reuse) {
}