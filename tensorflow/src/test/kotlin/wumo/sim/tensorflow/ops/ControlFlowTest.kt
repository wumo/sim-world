package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.tensorflow.tf

class ControlFlowTest {
  @Test
  fun cond() {
    val x = tf.const(0, name = "x")
    val y = tf.const(1, name = "y")
    val z = tf.cond(tf._less(x, y), { x * 17 }, { y * 23 })
    tf.printGraph()
  }
}