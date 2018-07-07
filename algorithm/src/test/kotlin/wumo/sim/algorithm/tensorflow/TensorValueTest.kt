package wumo.sim.algorithm.tensorflow

import wumo.sim.algorithm.tensorflow.ops.BaseTest

import org.junit.Assert.*
import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.util.helpers.f
import wumo.sim.algorithm.util.x

class TensorValueTest : BaseTest() {
  @Test
  fun `get set`() {
    val t = TensorValue.create(2 x 2, f(1f, 2f, 3f, 4f))
    println(t)
    t[0, 0] = 3f
    println(t)
    val t2 = TensorValue.wrap<Float>(t.c_tensor)
    println(t2)
    val t3 = TensorValue.create(1)
    println(t3)
    val a = tf.const(t3)
    printGraph()
    tf.session {
      a.eval()
    }
  }
}