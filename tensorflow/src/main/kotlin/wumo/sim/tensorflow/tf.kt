package wumo.sim.tensorflow

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import wumo.sim.tensorflow.ops.ops

object tf : ops.API {
  init {
    Loader.load(tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
  }
}