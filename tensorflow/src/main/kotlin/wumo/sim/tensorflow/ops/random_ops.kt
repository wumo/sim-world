package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.tensorflow.TF
import wumo.sim.tensorflow.is_integer
import wumo.sim.tensorflow.ops.gen.*
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape
import wumo.sim.util.scalarDimension

fun TF.random_normal(shape: Output, dtype: Int = DT_FLOAT,
                     mean: Float = 0f, stddev: Output,
                     name: String = "random_normal"): Output {
  name_scope(name) {
    val mean_t = const(mean, name = "mean")
    val rnd = randomStandardNormal(shape, dtype)
    val mul = rnd * stddev
    return add(mul, mean_t, ctxNs.scopeName)
  }
}

fun TF.random_normal(shape: Output, dtype: Int = DT_FLOAT,
                     mean: Float = 0f, stddev: Float = 1f,
                     name: String = "random_normal"): Output {
  name_scope(name) {
    val mean_t = const(mean, name = "mean")
    val stddev_t = const(stddev)
    val rnd = randomStandardNormal(shape, dtype)
    val mul = rnd * stddev_t
    return add(mul, mean_t, ctxNs.scopeName)
  }
}

fun TF.random_normal(shape: Output, dtype: Int = DT_FLOAT,
                     mean: Output, stddev: Output,
                     name: String = "RandomStandardNormal"): Output {
  name_scope(name) {
    val rnd = randomStandardNormal(shape, dtype)
    val mul = rnd * stddev
    return add(mul, mean, ctxNs.scopeName)
  }
}

fun TF.random_uniform(shape: Output, dtype: Int = DT_FLOAT,
                      min: Number, max: Number,
                      name: String = "random_uniform"): Output {
  name_scope(name) {
    val minval = const(scalarDimension, dtype, min, name = "min")
    val maxval = const(scalarDimension, dtype, max, name = "max")
    if (dtype.is_integer)
      return randomUniformInt(shape, minval, maxval, name = ctxNs.scopeName)
    val rand = randomUniform(shape, dtype)
    return add(rand * (maxval - minval), minval, ctxNs.scopeName)
  }
}

fun TF.random_uniform(shape: Shape,
                      min: Float, max: Float,
                      name: String = "random_uniform"): Output {
  name_scope(name) {
    val shape_t = tf.const(shape.asIntArray()!!, "shape")
    val minval = const(min, "min")
    val maxval = const(max, "max")
    val rand = randomUniform(shape_t, DT_FLOAT)
    
    return add(rand * (maxval - minval), minval, ctxNs.scopeName)
  }
}

fun TF.truncatedNormal(shape: Output, mean: Float = 0f, stddev: Float = 1f, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Output {
  name_scope(name) {
    val mean_t = const(mean, name = "mean")
    val stddev_t = const(stddev, name = "stddev")
    val rnd = truncatedNormal(shape, dtype, name = name)
    val mul = rnd * stddev_t
    return add(mul, mean_t, name = ctxNs.scopeName)
  }
}