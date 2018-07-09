package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.util.helpers.println

open class BaseTest {
  fun printGraph() {
    tf.debugString().println()
  }
}