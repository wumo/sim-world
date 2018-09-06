package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.Shape
import wumo.sim.util.f

class Math_opsKtTest : BaseTest() {
  
  @Test
  fun matmul() {
    val A = tf.const(Shape(1, 4), f(1f, 2f, 3f, 4f))
    val B = tf.const(Shape(4, 1), f(1f, 2f, 3f, 4f))
    val C = tf.matMul(A, B)
    
    printGraph()
    tf.session {
      C.eval()
    }
  }
  
  @Test
  fun addN() {
    val a = tf.const(Shape(2, 2), 1f, name = "a")
    val b = tf.const(Shape(2, 2), 2f, name = "b")
    val c = tf.const(Shape(2, 2), 3f, name = "c")
    val d = tf.addN(listOf(a, b, c), name = "addn")
    printGraph()
    tf.session {
      d.eval()
    }
  }
  
  @Test
  fun cast() {
    val a = tf.const(Shape(2, 2), 2.5f, "a")
    val b = tf.cast(a, INT32, "b")
    printGraph()
    tf.session {
      b.eval()
    }
  }
  
  @Test
  fun argmax() {
    val A = tf.const(Shape(3, 3), 1f)
    val B = tf.argmax(A, 1)
    tf.session {
      B.eval()
    }
  }
}