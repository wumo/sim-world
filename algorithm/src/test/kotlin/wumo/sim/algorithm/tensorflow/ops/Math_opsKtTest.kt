package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_INT32
import org.junit.Test

import org.junit.Assert.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.util.helpers.f
import wumo.sim.algorithm.util.x

class Math_opsKtTest : BaseTest() {
  
  @Test
  fun matmul() {
    val A = tf.const(1 x 4, f(1f, 2f, 3f, 4f))
    val B = tf.const(4 x 1, f(1f, 2f, 3f, 4f))
    val C = tf.matmul(A, B)
    printGraph()
    tf.session {
      C.eval()
    }
  }
  
  @Test
  fun addN() {
    val a = tf.const(2 x 2, 1f, name = "a")
    val b = tf.const(2 x 2, 2f, name = "b")
    val c = tf.const(2 x 2, 3f, name = "c")
    val d = tf.addN(a, b, c, name = "addn")
    printGraph()
    tf.session {
      d.eval()
    }
  }
  
  @Test
  fun cast() {
    val a = tf.const(2 x 2, 2.5f, "a")
    val b = tf.cast(a, DT_INT32, "b")
    printGraph()
    tf.session {
      b.eval()
    }
  }
}