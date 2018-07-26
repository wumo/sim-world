package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.is_integer
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.Dimension
import wumo.sim.util.scalarDimension

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     name: String = "RandomStandardNormal"): Tensor {
  name_scope(name) {
    val p = g.nodeBuilder("RandomStandardNormal", ctxNs.fullName)
        .attrType("dtype", dtype)
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
  name_scope(name) {
    val mean_t = tf.const(mean)
    val stddev_t = tf.const(stddev)
    val rnd = random_normal(shape, dtype)
    val mul = rnd * stddev_t
    return add(mul, mean_t, ctxNs.scopeNameForOnce())
  }
}

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     mean: Tensor, stddev: Tensor,
                     name: String = "RandomStandardNormal"): Tensor {
  name_scope(name) {
    val rnd = random_normal(shape, dtype)
    val mul = rnd * stddev
    return add(mul, mean, ctxNs.scopeNameForOnce())
  }
}

fun TF.random_uniform(shape: Tensor, dtype: Int = DT_FLOAT,
                      min: Number, max: Number,
                      name: String = "RandomUniform"): Tensor {
  name_scope(name) {
    val minval = const(scalarDimension, dtype, min, name = "min")
    val maxval = const(scalarDimension, dtype, max, name = "max")
    if (dtype.is_integer)
      return random_uniform_int(shape, minval, maxval)
    val rand = random_uniform(shape, dtype)
    return add(rand * (maxval - minval), minval, ctxNs.scopeNameForOnce())
  }
}

fun TF.random_uniform_int(shape: Tensor, minval: Tensor, maxval: Tensor,
                          name: String = "RandomUniformInt"): Tensor {
  name_scope(name) {
    val p = g.nodeBuilder("RandomUniformInt", ctxNs.fullName)
        .addInput(shape)
        .addInput(minval)
        .addInput(maxval)
        .build()
    return Tensor(p, 0)
  }
}

fun TF.random_uniform(shape: Tensor,
                      dtype: Int = DT_FLOAT,
                      name: String = "RandomUniform"): Tensor {
  name_scope(name) {
    val p = g.nodeBuilder("RandomUniform", ctxNs.fullName)
        .attrType("dtype", dtype)
        .addInput(shape)
        .build()
    return Tensor(p, 0)
  }
}

fun TF.random_uniform(shape: Dimension,
                      dtype: Int = DT_FLOAT,
                      name: String = "RandomUniform"): Tensor {
  name_scope(name) {
    val p = g.nodeBuilder("RandomUniform", ctxNs.fullName)
        .attrType("dtype", dtype)
        .addInput(const(shape.asIntArray(), "shape"))
        .build()
    return Tensor(p, 0)
  }
}

fun TF.random_uniform(shape: Dimension,
                      min: Float, max: Float,
                      name: String = "RandomUniform"): Tensor {
  name_scope(name) {
    val rand = random_uniform(shape, DT_FLOAT)
    val minval = const(min, "min")
    val maxval = const(max, "max")
    return add(rand * (maxval - minval), minval, ctxNs.scopeNameForOnce())
  }
}

fun TF.truncatedNormal(shape: Tensor, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Tensor {
  val op = g.nodeBuilder("TruncatedNormal", ctxNs.getUniqueFullName(name))
      .addInput(shape)
      .attrType("dtype", dtype)
      .build()
  return Tensor(op, 0)
}

fun TF.truncatedNormal(shape: Tensor, mean: Float = 0f, stddev: Float = 1f, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Tensor {
  tf.name_scope(name) {
    val mean_t = tf.const(mean, name = "mean")
    val stddev_t = tf.const(stddev, name = "stddev")
    val rnd = tf.truncatedNormal(shape, dtype, name)
    val mul = rnd * stddev_t
    return tf.add(mul, mean_t, name = ctxNs.scopeNameForOnce())
  }
}