package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.ops.gen.gen_random_ops
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape
import wumo.sim.util.scalarDimension

object random_ops {
  interface API : gen_random_ops {
    
    fun randomNormal(shape: Output, dtype: DataType<*> = FLOAT,
                     mean: Float = 0f, stddev: Output,
                     name: String = "randomNormal"): Output =
        tf.nameScope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val rnd = super.randomStandardNormal(shape, dtype, 0L, 0L, "RandomStandardNormal")
          val mul = rnd * stddev
          tf.add(mul, mean_t, tf.currentNameScope)
        }
    
    fun randomNormal(shape: Output, dtype: DataType<*> = FLOAT,
                     mean: Float = 0f, stddev: Float = 1f,
                     name: String = "randomNormal"): Output =
        tf.nameScope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val stddev_t = tf.const(stddev)
          val rnd = super.randomStandardNormal(shape, dtype, 0L, 0L, "RandomStandardNormal")
          val mul = rnd * stddev_t
          tf.add(mul, mean_t, tf.currentNameScope)
        }
    
    fun randomNormal(shape: Output, dtype: DataType<*> = FLOAT,
                     mean: Output, stddev: Output,
                     name: String = "RandomStandardNormal"): Output =
        tf.nameScope(name) {
          val rnd = super.randomStandardNormal(shape, dtype, 0L, 0L, "RandomStandardNormal")
          val mul = rnd * stddev
          tf.add(mul, mean, tf.currentNameScope)
        }
    
    fun randomUniform(shape: Output, dtype: DataType<*> = FLOAT,
                      min: Number, max: Number,
                      name: String = "randomUniform"): Output =
        tf.nameScope(name) {
          val minval = tf.const(scalarDimension, dtype, min, name = "min")
          val maxval = tf.const(scalarDimension, dtype, max, name = "max")
          if (dtype.isInteger)
            super.randomUniformInt(shape, minval, maxval, 0L, 0L, name = tf.currentNameScope)
          else {
            val rand = super.randomUniform(shape, dtype, 0L, 0L, "RandomUniform")
            tf.add(rand * (maxval - minval), minval, tf.currentNameScope)
          }
        }
    
    fun randomUniform(shape: Shape,
                      min: Float, max: Float,
                      name: String = "randomUniform"): Output =
        tf.nameScope(name) {
          val shape_t = tf.const(shape.asIntArray()!!, "shape")
          val minval = tf.const(min, "min")
          val maxval = tf.const(max, "max")
          val rand = super.randomUniform(shape_t, FLOAT, 0L, 0L, "RandomUniform")
          
          tf.add(rand * (maxval - minval), minval, tf.currentNameScope)
        }
    
    fun truncatedNormal(shape: Output, mean: Float = 0f, stddev: Float = 1f, dtype: DataType<*> = FLOAT, name: String = "truncated_normal"): Output =
        tf.nameScope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val stddev_t = tf.const(stddev, name = "stddev")
          val rnd = super.truncatedNormal(shape, dtype, 0L, 0L, name = name)
          val mul = rnd * stddev_t
          tf.add(mul, mean_t, name = tf.currentNameScope)
        }
  }
}