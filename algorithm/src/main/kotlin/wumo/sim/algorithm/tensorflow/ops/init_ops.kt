package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Dimension

class Initializer(val dtype: Int, val name: String,
                  val init: (Dimension, Int, String) -> Tensor) {
  operator fun invoke(shape: Dimension, dtype: Int = this.dtype, name: String = this.name) =
      tf.name_scope(name) { init(shape, dtype, tf.ctxNs.scopeName) }
}

fun TF.zeros_initializer(dtype: Int = DT_FLOAT) = Initializer(dtype, "zeros_initializer") { shape, dtype, name ->
  zeros(shape, dtype, "zeros")
}

fun TF.ones_initializer(dtype: Int = DT_FLOAT) = Initializer(dtype, "oness_initializer") { shape, dtype, name ->
  ones(shape, dtype, "ones")
}

fun TF.constant_initializer(value: Any, dtype: Int = DT_FLOAT) = Initializer(dtype, "const_initializer") { shape, dtype, name ->
  const(shape, dtype, value, name = "Const")
}

fun TF.random_uniform_initializer() {
}

fun TF.random_normal_initializer() {
}

fun TF.identity_initializer() {
}

