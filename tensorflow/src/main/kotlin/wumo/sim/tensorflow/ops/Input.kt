package wumo.sim.tensorflow.ops

class Input(val op: Op?, val value_index: Int) {
  val name: String by lazy { "${op!!.name}:$value_index" }
}