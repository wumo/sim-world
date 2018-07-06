package wumo.sim.algorithm.util.cpp_api.layers

import org.bytedeco.javacpp.tensorflow.*

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
            dtype: Int = -1,
            name: String = "", scope: Scope? = null) : Layer(trainable = trainable,
                                                             activity_reqularizer =
                                                             activity_regularizer,
                                                             dtype = dtype,
                                                             name = name, scope = scope) {
  lateinit var input_spec: Any
  
  fun build() {
  
  }
  
  
  operator fun invoke(inputs: Output, shape: Dimension): Any {
    
    TODO("not implemented")
  }
}