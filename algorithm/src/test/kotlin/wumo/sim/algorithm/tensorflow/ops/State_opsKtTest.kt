package wumo.sim.algorithm.tensorflow.ops

import org.junit.Test

import org.junit.Assert.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.util.helpers.a
import wumo.sim.algorithm.util.helpers.f
import wumo.sim.algorithm.util.x

class State_opsKtTest : BaseTest() {
  
  @Test
  fun variable() {
    val a1 = tf.variable(4 x 4 x 2, 1)
    val b = tf.variable(f(1f, 2f, 3f, 4f))
    val c = tf.variable(2 x 2, a("1", "2", "a", "b"))
    val init = tf.global_variable_initializer()
    printGraph()
    tf.session {
      init.run()
      a1.eval()
      b.eval()
      c.eval()
    }
  }
}