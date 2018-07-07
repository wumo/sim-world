package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.tensorflow.*
import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.BaseTest

import org.junit.Assert.*
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.util.helpers.a
import wumo.sim.algorithm.util.helpers.f
import wumo.sim.algorithm.util.x

class SessionTest : BaseTest() {
  
  @Test
  fun eval() {
    val c = tf.const(2 x 2 x 2, 6f)
    val d = tf.const(2 x 2, f(1f, 2f, 3f, 4f), name = "d")
    val e = tf.const(2 x 2, a("hello", "tensorflow", "and", "you"), name = "e")
    val f = tf.const(a("hello", "tensorflow", "and", "you"), name = "f")
    printGraph()
    tf.session {
      a(c, d, e, f).eval()
    }
  }
}