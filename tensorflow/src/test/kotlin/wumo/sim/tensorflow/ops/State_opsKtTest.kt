package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.f

class State_opsKtTest : BaseTest() {
  
  @Test
  fun `variable def`() {
    val a1 = tf.variable(Shape(4, 4, 2), 1, name = "a")
    val b = tf.variable(f(1f, 2f, 3f, 4f), name = "b")
    val init = tf.globalVariablesInitializer()
    printGraph()
  }
  
  @Test
  fun variable() {
    val a = tf.variable(Shape(4, 4, 2), 1, name = "a")
    val b = tf.variable(f(1f, 2f, 3f, 4f), name = "b")
    val init = tf.globalVariablesInitializer()
    printGraph()
    tf.session {
      init.run()
      a.eval()
      b.eval()
    }
  }
  
  @Test
  fun `variable depend on variable`() {
    val a = tf.variable(1, "a")
    val b = tf.variable(a, name = "b")
    val init = tf.globalVariablesInitializer()
    printGraph()
    tf.session {
      b.initializer.run()
      a.initializer.run()
      b.eval()
    }
  }
}