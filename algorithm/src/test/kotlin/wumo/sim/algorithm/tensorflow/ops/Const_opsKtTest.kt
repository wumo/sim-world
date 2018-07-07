package wumo.sim.algorithm.tensorflow.ops

import org.junit.After
import org.junit.Before
import org.junit.Test

import org.junit.Assert.*
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.util.helpers.println
import wumo.sim.algorithm.util.x

class Const_opsKtTest {
  lateinit var tf: TF
  @Before
  fun setUp() {
    tf = TF()
  }
  
  @Test
  fun const1() {
    val a = tf.const(1f)
    val b = tf.const(2f, name = "b")
    val c = tf.const(2 x 2, 6L, name = "c")
    val d = tf.const(2 x 2, "hello", name = "d")
    tf.debugString().println()
  }
  
  @Test
  fun const2() {
    val c = tf.const(2 x 2 x 2, 6f, name = "c")
    val d = tf.const(2 x 2, floatArrayOf(1f, 2f, 3f, 4f), name = "d")
    val e = tf.const(2 x 2, arrayOf("hello", "tensorflow", "and", "you"), name = "e")
    val f = tf.const1D("hello", "tensorflow", "and", "you", name = "f")
    tf.debugString().println()
    tf.session {
      c.eval()
      d.eval()
      e.eval()
      f.eval()
    }
  }
  
}