package wumo.sim.algorithm.tensorflow

import wumo.sim.algorithm.tensorflow.ops.BaseTest

import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.util.f
import wumo.sim.util.x

class TensorValueTest {
  @Test
  fun `get set`() {
    tf
    val t = TensorValue(2 x 2, f(1f, 2f, 3f, 4f))
    println(t)
    t[0, 0] = 3f
    println(t)
    val t2 = TensorValue<Float>(t.c_tensor)
    println(t2)
    val t3 = TensorValue(1)
    println(t3)
    val a = tf.const(t3)
    tf.printGraph()
    tf.session {
      a.eval()
    }
  }
}