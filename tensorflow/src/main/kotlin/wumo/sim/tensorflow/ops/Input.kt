package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.ops.Op

class Input(val op: Op?, val value_index: Int) {
  val name: String by lazy { "${op!!.name}:$value_index" }
}