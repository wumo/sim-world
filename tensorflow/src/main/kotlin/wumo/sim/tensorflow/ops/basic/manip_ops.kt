package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_manip_ops

object manip_ops {
  interface API {
    fun roll(input: Output, shift: Output, axis: Output, name: String = "Roll"): Output {
      return gen_manip_ops.roll(input, shift, axis, name)
    }
  }
}