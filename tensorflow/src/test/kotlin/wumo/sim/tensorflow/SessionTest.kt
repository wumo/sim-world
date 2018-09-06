package wumo.sim.tensorflow

import org.junit.Test
import wumo.sim.tensorflow.ops.BaseTest
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.f

class SessionTest : BaseTest() {
  
  @Test
  fun eval() {
    val c = tf.const(Shape(2, 2, 2), 6f)
    val d = tf.const(Shape(2, 2), f(1f, 2f, 3f, 4f), name = "d")
    val e = tf.const(Shape(2, 2), a("hello", "tensorflow", "and", "you"), name = "e")
    val f = tf.const(a("hello", "tensorflow", "and", "you"), name = "f")
    printGraph()
    tf.session {
      a(c, d, e, f).eval()
    }
  }
}