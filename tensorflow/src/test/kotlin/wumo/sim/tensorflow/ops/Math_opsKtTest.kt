package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.DT_INT32
import org.junit.Test
import wumo.sim.tensorflow.ops.gen.addN
import wumo.sim.tensorflow.ops.gen.matMul

import wumo.sim.tensorflow.tf
import wumo.sim.util.f
import wumo.sim.util.println
import wumo.sim.util.x

class Math_opsKtTest : BaseTest() {
  
  @Test
  fun matmul() {
    val A = tf.const(1 x 4, f(1f, 2f, 3f, 4f))
    val B = tf.const(4 x 1, f(1f, 2f, 3f, 4f))
    val C = tf.matMul(A, B)
    val node = tensorflow.Node(C.op!!.c_op)
    node.DebugString().string.println()

//    printGraph()
//    tf.session {
//      C.eval()
//    }
  }
  
  @Test
  fun addN() {
    val a = tf.const(2 x 2, 1f, name = "a")
    val b = tf.const(2 x 2, 2f, name = "b")
    val c = tf.const(2 x 2, 3f, name = "c")
    val d = tf.addN(arrayOf(a, b, c), name = "addn")
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
  
  @Test
  fun argmax() {
    val A = tf.const(3 x 3, 1f)
    val B = tf.argmax(A, 1)
    tf.session {
      B.eval()
    }
  }
}