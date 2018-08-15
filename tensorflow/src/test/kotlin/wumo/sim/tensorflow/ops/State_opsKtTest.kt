package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.f

class State_opsKtTest : BaseTest() {
  
  @Test
  fun `variable def`() {
    val a1 = tf.variable(Shape(4, 4, 2), 1)
    val b = tf.variable(f(1f, 2f, 3f, 4f))
    val c = tf.variable(Shape(2, 2), a("1", "2", "a", "b"))
    printGraph()
  }
  
  @Test
  fun variable() {
    val a1 = tf.variable(Shape(4, 4, 2), 1)
    val b = tf.variable(f(1f, 2f, 3f, 4f))
    val c = tf.variable(Shape(2, 2), a("1", "2", "a", "b"))
    val init = tf.global_variable_initializer()
    printGraph()
    tf.session {
      init.run()
      a1.eval()
      b.eval()
      c.eval()
    }
  }
  
  @Test
  fun `variable depend on variable`() {
//    val a = tf.variable(2 x 2, 1f, "a")
//    val b = tf.variable(a, name = "b")
////    val init = tf.global_variable_initializer()
//    printGraph()
//    tf.session {
//      b.initializer_op.op!!.run()
//      a.initializer_op.op!!.run()
//      b.eval()
//    }
  }
}