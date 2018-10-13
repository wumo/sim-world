package wumo.sim.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Tensor.newTensor
import org.bytedeco.javacpp.tensorflow.DT_UINT8
import org.junit.Test
import wumo.sim.tensorflow.ops.BaseTest
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.f
import wumo.sim.util.native

class SessionTest : BaseTest() {
  
  @Test
  fun eval() {
    val c = tf.const(Shape(2, 2, 2), 6f)
    val d = tf.const(Shape(2, 2), f(1f, 2f, 3f, 4f), name = "d")
    val e = tf.const(Shape(2, 2), a("hello", "tensorflow", "and", "you"), name = "e")
    val f = tf.const(a("hello", "tensorflow", "and", "you"), name = "f")
    printGraph()
    tf.session {
      listOf(c, d, e, f).eval()
    }
  }
  
  @Test
  fun testDeallocate() {
    tf
    while (true)
      native {
        val data = BytePointer(100L)
        val t = newTensor(DT_UINT8, longArrayOf(data.limit()), data)
      }
  }
}