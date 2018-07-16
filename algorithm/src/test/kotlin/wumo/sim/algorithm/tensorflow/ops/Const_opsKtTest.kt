package wumo.sim.algorithm.tensorflow.ops

import org.junit.Test

import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.a
import wumo.sim.util.f
import wumo.sim.util.x

class Const_opsKtTest : BaseTest() {
  
  @Test
  fun const1() {
    val a = tf.const(1f)
    val b = tf.const(2f, name = "b")
    val c = tf.const(2 x 2, 6L, name = "c")
    val d = tf.const(2 x 2, "hello", name = "d")
    printGraph()
  }
  
  @Test
  fun const2() {
    val c = tf.const(2 x 2 x 2, 6f)
    val d = tf.const(2 x 2, f(1f, 2f, 3f, 4f), name = "d")
    val e = tf.const(2 x 2, a("hello", "tensorflow", "and", "you"), name = "e")
    val f = tf.const(a("hello", "tensorflow", "and", "you"), name = "f")
    printGraph()
    tf.session {
      c.eval()
      d.eval()
      e.eval()
      f.eval()
    }
  }
  
}