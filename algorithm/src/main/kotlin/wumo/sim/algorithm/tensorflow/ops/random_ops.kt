package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.util.Dimension
import wumo.sim.util.scalarDimension

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     name: String = "RandomStandardNormal") =
    name_scope(name) {
      naryOp("RandomStandardNormal", shape, name = name) {
        attrType("dtype", dtype)
      }
    }

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     mean: Float = 0f, stddev: Tensor,
                     name: String = "random_normal"): Tensor {
  name_scope(name) {
    val mean_t = tf.const(mean, name = "mean")
    val rnd = random_normal(shape, dtype)
    val mul = rnd * stddev
    return add(mul, mean_t, ctxNs.scopeName)
  }
}

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     mean: Float = 0f, stddev: Float = 1f,
                     name: String = "random_normal"): Tensor {
  name_scope(name) {
    val mean_t = tf.const(mean, name = "mean")
    val stddev_t = tf.const(stddev)
    val rnd = random_normal(shape, dtype)
    val mul = rnd * stddev_t
    return add(mul, mean_t, ctxNs.scopeName)
  }
}

fun TF.random_normal(shape: Tensor, dtype: Int = DT_FLOAT,
                     mean: Tensor, stddev: Tensor,
                     name: String = "RandomStandardNormal"): Tensor {
  name_scope(name) {
    val rnd = random_normal(shape, dtype)
    val mul = rnd * stddev
    return add(mul, mean, ctxNs.scopeName)
  }
}

fun TF.random_uniform(shape: Tensor, dtype: Int = DT_FLOAT,
                      min: Number, max: Number,
                      name: String = "random_uniform"): Tensor {
  name_scope(name) {
    val minval = const(scalarDimension, dtype, min, name = "min")
    val maxval = const(scalarDimension, dtype, max, name = "max")
    if (dtype.is_integer)
      return random_uniform_int(shape, minval, maxval, ctxNs.scopeName)
    val rand = _random_uniform(shape, dtype)
    return add(rand * (maxval - minval), minval, ctxNs.scopeName)
  }
}

fun TF.random_uniform_int(shape: Tensor, minval: Tensor, maxval: Tensor,
                          name: String = "RandomUniformInt") =
    naryOp("RandomUniformInt", shape, minval, maxval, name = name)

fun TF._random_uniform(shape: Tensor,
                       dtype: Int = DT_FLOAT,
                       name: String = "RandomUniform") =
    naryOp("RandomUniform", shape, name = name) {
      attrType("dtype", dtype)
    }

fun TF._random_uniform(shape: Dimension,
                       dtype: Int = DT_FLOAT,
                       name: String = "RandomUniform") =
    name_scope(name) {
      val shape = const(shape.asIntArray(), "shape")
      naryOp("RandomUniform", shape, name = ctxNs.scopeName) {
        attrType("dtype", dtype)
      }
    }

fun TF.random_uniform(shape: Dimension,
                      min: Float, max: Float,
                      name: String = "random_uniform"): Tensor {
  name_scope(name) {
    val shape_t = tf.const(shape.asIntArray(), "shape")
    val minval = const(min, "min")
    val maxval = const(max, "max")
    val rand = _random_uniform(shape_t, DT_FLOAT)
    
    return add(rand * (maxval - minval), minval, ctxNs.scopeName)
  }
}

fun TF.truncatedNormal(shape: Tensor, dtype: Int = DT_FLOAT, name: String = "truncated_normal") =
    name_scope(name) {
      naryOp("TruncatedNormal", shape, name = name) {
        attrType("dtype", dtype)
      }
    }

fun TF.truncatedNormal(shape: Tensor, mean: Float = 0f, stddev: Float = 1f, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Tensor {
  tf.name_scope(name) {
    val mean_t = tf.const(mean, name = "mean")
    val stddev_t = tf.const(stddev, name = "stddev")
    val rnd = tf.truncatedNormal(shape, dtype, name)
    val mul = rnd * stddev_t
    return tf.add(mul, mean_t, name = ctxNs.scopeName)
  }
}