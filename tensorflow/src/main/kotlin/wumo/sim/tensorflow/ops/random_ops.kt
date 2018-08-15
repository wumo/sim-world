package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape
import wumo.sim.util.scalarDimension

object random_ops {
  interface API {
    
    fun random_normal(shape: Output, dtype: DataType<*> = FLOAT,
                      mean: Float = 0f, stddev: Output,
                      name: String = "random_normal"): Output =
        tf.name_scope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val rnd = tf._randomStandardNormal(shape, dtype)
          val mul = rnd * stddev
          tf._add(mul, mean_t, tf.currentNameScope)
        }
    
    fun random_normal(shape: Output, dtype: DataType<*> = FLOAT,
                      mean: Float = 0f, stddev: Float = 1f,
                      name: String = "random_normal"): Output =
        tf.name_scope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val stddev_t = tf.const(stddev)
          val rnd = tf._randomStandardNormal(shape, dtype)
          val mul = rnd * stddev_t
          tf._add(mul, mean_t, tf.currentNameScope)
        }
    
    fun random_normal(shape: Output, dtype: DataType<*> = FLOAT,
                      mean: Output, stddev: Output,
                      name: String = "RandomStandardNormal"): Output =
        tf.name_scope(name) {
          val rnd = tf._randomStandardNormal(shape, dtype)
          val mul = rnd * stddev
          tf._add(mul, mean, tf.currentNameScope)
        }
    
    fun random_uniform(shape: Output, dtype: DataType<*> = FLOAT,
                       min: Number, max: Number,
                       name: String = "random_uniform"): Output =
        tf.name_scope(name) {
          val minval = tf.const(scalarDimension, dtype, min, name = "min")
          val maxval = tf.const(scalarDimension, dtype, max, name = "max")
          if (dtype.isInteger)
            tf._randomUniformInt(shape, minval, maxval, name = tf.currentNameScope)
          else {
            val rand = tf._randomUniform(shape, dtype)
            tf._add(rand * (maxval - minval), minval, tf.currentNameScope)
          }
        }
    
    fun random_uniform(shape: Shape,
                       min: Float, max: Float,
                       name: String = "random_uniform"): Output =
        tf.name_scope(name) {
          val shape_t = tf.const(shape.asIntArray()!!, "shape")
          val minval = tf.const(min, "min")
          val maxval = tf.const(max, "max")
          val rand = tf._randomUniform(shape_t, FLOAT)
          
          tf._add(rand * (maxval - minval), minval, tf.currentNameScope)
        }
    
    fun truncatedNormal(shape: Output, mean: Float = 0f, stddev: Float = 1f, dtype: DataType<*> = FLOAT, name: String = "truncated_normal"): Output =
        tf.name_scope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val stddev_t = tf.const(stddev, name = "stddev")
          val rnd = tf._truncatedNormal(shape, dtype, name = name)
          val mul = rnd * stddev_t
          tf._add(mul, mean_t, name = tf.currentNameScope)
        }
  }
}