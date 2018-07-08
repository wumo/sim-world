package wumo.sim.algorithm.tensorflow.layers

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.util.Dimension

class Dense(val units: Int,
            val activation: Any? = null,
            val use_bias: Boolean = true,
            val kernel_initializer: Any? = null,
            val bias_initializer: Any? = null,
            val kernel_regularizer: Any? = null,
            val bias_regularizer: Any? = null,
            activity_regularizer: Any? = null,
            val kernel_constraint: Any? = null,
            val bias_constraint: Any? = null,
            trainable: Boolean = true,
            dtype: Int = 0,
            name: String = "") : Layer(trainable = trainable,
                                       activity_reqularizer =
                                       activity_regularizer,
                                       dtype = dtype,
                                       name = name) {
  lateinit var input_spec: Any
  
  override fun build(input_shape: Dimension) {
    if (input_shape[-1] == -1)
      throw IllegalArgumentException("")
  }
  
  override fun call(input: Tensor): Tensor {
    TODO()
  }
  
}