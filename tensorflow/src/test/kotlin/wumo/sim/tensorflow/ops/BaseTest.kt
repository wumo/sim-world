package wumo.sim.tensorflow.ops

import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.println

open class BaseTest {
  fun printGraph() {
    tf.debugString().println()
  }
}