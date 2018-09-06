package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_bitwise_ops

object bitwise_ops {
  interface API {
    fun bitwiseAnd(x: Output, y: Output, name: String = "BitwiseAnd"): Output {
      return gen_bitwise_ops.bitwiseAnd(x, y, name)
    }
    
    fun bitwiseOr(x: Output, y: Output, name: String = "BitwiseOr"): Output {
      return gen_bitwise_ops.bitwiseOr(x, y, name)
    }
    
    fun bitwiseXor(x: Output, y: Output, name: String = "BitwiseXor"): Output {
      return gen_bitwise_ops.bitwiseXor(x, y, name)
    }
    
    fun invert(x: Output, name: String = "Invert"): Output {
      return gen_bitwise_ops.invert(x, name)
    }
    
    fun leftShift(x: Output, y: Output, name: String = "LeftShift"): Output {
      return gen_bitwise_ops.leftShift(x, y, name)
    }
    
    fun populationCount(x: Output, name: String = "PopulationCount"): Output {
      return gen_bitwise_ops.populationCount(x, name)
    }
    
    fun rightShift(x: Output, y: Output, name: String = "RightShift"): Output {
      return gen_bitwise_ops.rightShift(x, y, name)
    }
  }
}