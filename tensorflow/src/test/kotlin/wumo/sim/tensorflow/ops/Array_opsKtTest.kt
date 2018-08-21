package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.tensorflow.tf
import wumo.sim.util.f
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.x

class Array_opsKtTest : BaseTest() {
  
  @Test
  fun identity() {
  }
  
  @Test
  fun placeholder() {
    val p = tf.placeholder(2 x 2)
    val a = tf.variable(p)
    val init = tf.globalVariablesInitializer()
    printGraph()
    tf.session {
      init.run(p to NDArray(2 x 2, f(1f, 2f, 3f, 4f)))
      a.eval()
    }
  }
  
  @Test
  fun `one hot encoding`() {
//    val indices = tf.const(i(0, 2, -1, 1), name = "indices")
//    val onehot = tf.one_hot_encoding(indices, 3, 5f, 0f)
//
//    println(tf.debugString())
//    tf.session {
//      val result = eval<Float>(onehot)
//      println(result)
//    }
  }
}