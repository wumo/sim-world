package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.util.Dimension

class Initializer(val dtype: Int, val name: String,
                  val init: (Dimension, Int, String) -> Tensor) {
  
  operator fun invoke(shape: Dimension, dtype: Int = this.dtype, name: String = this.name) =
      init(shape, dtype, name)
}

fun TF.zeros_initializer(dtype: Int = DT_FLOAT) = Initializer(dtype, "zeros_initializer") { shape, dtype, name ->
  zeros(shape, dtype, name)
}

fun TF.ones_initializer(dtype: Int = DT_FLOAT) = Initializer(dtype, "oness_initializer") { shape, dtype, name ->
  ones(shape, dtype, name)
}

fun TF.constant_initializer(value: Any, dtype: Int = DT_FLOAT) = Initializer(dtype, "const_initializer") { shape, dtype, name ->
  const(shape, dtype, value, name)
}

fun TF.random_uniform_initializer() {

}

fun TF.random_normal_initializer() {

}

fun TF.identity_initializer() {

}

