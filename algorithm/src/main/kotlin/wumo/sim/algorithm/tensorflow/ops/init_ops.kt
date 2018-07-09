package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.util.Dimension

fun TF.random_uniform(shape: Dimension,
                      dtype: Int = tensorflow.DT_FLOAT,
                      name: String = "RandomUniform"): Tensor {
  subscope(name) {
    val p = g.nodeBuilder("RandomUniform", parentName)
        .setAttrType("dtype", dtype)
        .addInput(const(shape.asIntArray(), "shape"))
        .build()
    return Tensor(p, 0, dtype)
  }
}

fun TF.random_uniform(shape: Dimension,
                      min: Float, max: Float,
                      name: String = "RandomUniform"): Tensor {
  subscope(name) {
    val rand = random_uniform(shape, tensorflow.DT_FLOAT)
    val min = const(min, "min")
    val max = const(max, "max")
//    return rand * (max - min) + min
    return add(rand * (max - min), min, borrowParentName())
  }
}

typealias Initializer = (Dimension, Int, String) -> Tensor

fun TF.zeros_initializer(): Initializer = { shape, dtype, name ->
  zeros(shape, dtype, name)
}

fun TF.ones_initializer(): Initializer = { shape, dtype, name ->
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

