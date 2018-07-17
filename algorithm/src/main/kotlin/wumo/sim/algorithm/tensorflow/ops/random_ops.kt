package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Dimension

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     name: String = "RandomStandardNormal"): Tensor {
  subscope(name) {
    val p = g.nodeBuilder("RandomStandardNormal", parentName)
        .setAttrType("dtype", dtype)
        .addInput(shape)
        .build()
    return Tensor(p, 0)
  }
}

fun TF.random_normal(shape: Dimension, dtype: Int = DT_FLOAT,
                     mean: Float = 0f, stddev: Float = 1f,
                     name: String = "RandomStandardNormal") =
    random_normal(tf.const(shape.asIntArray()), dtype, mean, stddev, name)

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     mean: Float = 0f, stddev: Float = 1f,
                     name: String = "RandomStandardNormal"): Tensor {
  subscope(name) {
    val mean_t = tf.const(mean)
    val stddev_t = tf.const(stddev)
    val rnd = random_normal(shape, dtype)
    val mul = rnd * stddev_t
    return add(mul, mean_t, parentName)
  }
}

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     mean: Tensor, stddev: Tensor,
                     name: String = "RandomStandardNormal"): Tensor {
  subscope(name) {
    val rnd = random_normal(shape, dtype)
    val mul = rnd * stddev
    return add(mul, mean, parentName)
  }
}

fun TF.random_uniform(shape: Tensor, dtype: Int = DT_FLOAT,
                      min: Number, max: Number,
                      name: String = "RandomUniform"): Tensor {
  TODO()
}

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

fun TF.truncatedNormal(shape: Tensor, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Tensor {
  val op = g.nodeBuilder("TruncatedNormal", ctx.getUniqueFullName(name))
      .addInput(shape)
      .setAttrType("dtype", dtype)
      .build()
  return Tensor(op, 0)
}

fun TF.truncatedNormal(shape: Tensor, mean: Float = 0f, stddev: Float = 1f, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Tensor {
  tf.subscope(name) {
    val mean_t = tf.const(mean, name = "mean")
    val stddev_t = tf.const(stddev, name = "stddev")
    val rnd = tf.truncatedNormal(shape, dtype, name)
    val mul = rnd * stddev_t
    return tf.add(mul, mean_t, name = name)
  }
}