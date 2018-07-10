package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.util.Dimension

fun TF.random_uniform(shape: Dimension,
                      dtype: Int = DT_FLOAT,
                      name: String = "RandomUniform"): Tensor {
  subscope(name) {
    val p = g.nodeBuilder("RandomUniform", parentName)
        .setAttrType("dtype", dtype)
        .addInput(const(shape.asIntArray(), "shape"))
        .build()
    return Tensor(p, 0)
  }
}

fun TF.random_uniform(shape: Dimension,
                      min: Float, max: Float,
                      name: String = "RandomUniform"): Tensor {
  subscope(name) {
    val rand = random_uniform(shape, DT_FLOAT)
    val min = const(min, "min")
    val max = const(max, "max")
//    return rand * (max - min) + min
    return add(rand * (max - min), min, borrowParentName())
  }
}

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

fun TF.constant_initializer() {

}

fun TF.random_uniform_initializer() {

}

fun TF.random_normal_initializer() {

}

fun TF.identity_initializer() {

}

