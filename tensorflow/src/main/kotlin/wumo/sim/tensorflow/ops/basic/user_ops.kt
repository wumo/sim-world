package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_user_ops

object user_ops {
  interface API {
    fun fact(name: String = "Fact"): Output {
      return gen_user_ops.fact(name)
    }
  }
}