package wumo.sim.algorithm.tensorflow.ops

import org.junit.Before
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.util.helpers.println

open class BaseTest {
  lateinit var tf: TF
  @Before
  fun setUp() {
    tf = TF()
  }
  
  fun printGraph() {
    tf.debugString().println()
  }
}