package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_stateless_random_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT64

object stateless_random_ops {
  interface API {
    fun statelessMultinomial(logits: Output, numSamples: Output, seed: Output, outputDtype: DataType<*> = INT64, name: String = "StatelessMultinomial"): Output {
      return gen_stateless_random_ops.statelessMultinomial(logits, numSamples, seed, outputDtype, name)
    }
    
    fun statelessRandomNormal(shape: Output, seed: Output, dtype: DataType<*> = FLOAT, name: String = "StatelessRandomNormal"): Output {
      return gen_stateless_random_ops.statelessRandomNormal(shape, seed, dtype, name)
    }
    
    fun statelessRandomUniform(shape: Output, seed: Output, dtype: DataType<*> = FLOAT, name: String = "StatelessRandomUniform"): Output {
      return gen_stateless_random_ops.statelessRandomUniform(shape, seed, dtype, name)
    }
    
    fun statelessTruncatedNormal(shape: Output, seed: Output, dtype: DataType<*> = FLOAT, name: String = "StatelessTruncatedNormal"): Output {
      return gen_stateless_random_ops.statelessTruncatedNormal(shape, seed, dtype, name)
    }
  }
}