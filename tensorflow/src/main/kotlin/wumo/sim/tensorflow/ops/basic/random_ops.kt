package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.framework.getSeed
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_random_ops
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape
import wumo.sim.util.scalarDimension

object random_ops {
  private val allowedTypes = setOf(FLOAT16, BFLOAT16, FLOAT, DOUBLE, INT32, INT64)
  
  interface API {
    fun multinomial(logits: Output, numSamples: Output, seed: Long = 0L, seed2: Long = 0L, outputDtype: DataType<*> = INT64, name: String = "Multinomial"): Output {
      return gen_random_ops.multinomial(logits, numSamples, seed, seed2, outputDtype, name)
    }
    
    fun parameterizedTruncatedNormal(shape: Output, means: Output, stdevs: Output, minvals: Output, maxvals: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "ParameterizedTruncatedNormal"): Output {
      return gen_random_ops.parameterizedTruncatedNormal(shape, means, stdevs, minvals, maxvals, seed, seed2, name)
    }
    
    fun randomGamma(shape: Output, alpha: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomGamma"): Output {
      return gen_random_ops.randomGamma(shape, alpha, seed, seed2, name)
    }
    
    fun randomGammaGrad(alpha: Output, sample: Output, name: String = "RandomGammaGrad"): Output {
      return gen_random_ops.randomGammaGrad(alpha, sample, name)
    }
    
    fun randomNormal(shape: Output, dtype: DataType<*> = FLOAT,
                     mean: Float = 0f, stddev: Output,
                     name: String = "randomNormal"): Output =
        tf.nameScope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val rnd = gen_random_ops.randomStandardNormal(shape, dtype, 0L, 0L, "RandomStandardNormal")
          val mul = rnd * stddev
          tf.add(mul, mean_t, tf.currentNameScope)
        }
    
    fun randomNormal(shape: Output, dtype: DataType<*> = FLOAT,
                     mean: Float = 0f, stddev: Float = 1f,
                     name: String = "randomNormal"): Output =
        tf.nameScope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val stddev_t = tf.const(stddev)
          val rnd = gen_random_ops.randomStandardNormal(shape, dtype, 0L, 0L, "RandomStandardNormal")
          val mul = rnd * stddev_t
          tf.add(mul, mean_t, tf.currentNameScope)
        }
    
    fun randomNormal(shape: Output, dtype: DataType<*> = FLOAT,
                     mean: Output, stddev: Output,
                     name: String = "RandomStandardNormal"): Output =
        tf.nameScope(name) {
          val rnd = gen_random_ops.randomStandardNormal(shape, dtype, 0L, 0L, "RandomStandardNormal")
          val mul = rnd * stddev
          tf.add(mul, mean, tf.currentNameScope)
        }
    
    fun randomPoisson(shape: Output, rate: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomPoisson"): Output {
      return gen_random_ops.randomPoisson(shape, rate, seed, seed2, name)
    }
    
    fun randomPoissonV2(shape: Output, rate: Output, seed: Long = 0L, seed2: Long = 0L, dtype: DataType<*> = INT64, name: String = "RandomPoissonV2"): Output {
      return gen_random_ops.randomPoissonV2(shape, rate, seed, seed2, dtype, name)
    }
    
    fun randomShuffle(value: Output, seed: Long = 0L, seed2: Long = 0L, name: String = "RandomShuffle"): Output {
      return gen_random_ops.randomShuffle(value, seed, seed2, name)
    }
    
    fun randomUniform(shape: Shape, min: Number, max: Number,
                      dtype: DataType<*> = FLOAT,
                      seed: Int? = null,
                      name: String = "randomUniform"): Output =
        randomUniform(shape.toOutput(), min, max, dtype, seed, name)
    
    fun randomUniform(shape: Output, min: Number, max: Number,
                      dtype: DataType<*> = FLOAT,
                      seed: Int? = null,
                      name: String = "randomUniform"): Output {
      require(dtype in allowedTypes) { "Invalid dtype$dtype" }
      
      return tf.nameScope(name, setOf(shape.op)) {
        val minval = tf.const(scalarDimension, dtype, min, name = "min")
        val maxval = tf.const(scalarDimension, dtype, max, name = "max")
        val (seed1, seed2) = getSeed(seed)
        if (dtype.isInteger)
          gen_random_ops.randomUniformInt(shape, minval, maxval,
                                          seed1?.toLong() ?: 0L,
                                          seed2?.toLong() ?: 0L,
                                          name = tf.currentNameScope)
        else {
          val rand = gen_random_ops.randomUniform(shape, dtype,
                                                  seed1?.toLong() ?: 0L,
                                                  seed2?.toLong() ?: 0L, "RandomUniform")
          tf.add(rand * (maxval - minval), minval, tf.currentNameScope)
        }
      }
    }
    
    fun truncatedNormal(shape: Output, mean: Float = 0f, stddev: Float = 1f, dtype: DataType<*> = FLOAT, name: String = "truncated_normal"): Output =
        tf.nameScope(name) {
          val mean_t = tf.const(mean, name = "mean")
          val stddev_t = tf.const(stddev, name = "stddev")
          val rnd = gen_random_ops.truncatedNormal(shape, dtype, 0L, 0L, name = name)
          val mul = rnd * stddev_t
          tf.add(mul, mean_t, name = tf.currentNameScope)
        }
  }
}