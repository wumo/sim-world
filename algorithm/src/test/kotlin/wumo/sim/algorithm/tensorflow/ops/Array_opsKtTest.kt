package wumo.sim.algorithm.tensorflow.ops

import org.junit.Test

import org.junit.Assert.*
import wumo.sim.algorithm.tensorflow.TensorValue
import wumo.sim.algorithm.tensorflow.contrib.one_hot_encoding
import wumo.sim.algorithm.util.helpers.f
import wumo.sim.algorithm.util.helpers.i
import wumo.sim.algorithm.util.x

class Array_opsKtTest : BaseTest() {
  
  @Test
  fun identity() {
  }
  
  @Test
  fun placeholder() {
    val p = tf.placeholder(2 x 2)
    val a = tf.variable(p)
    val init = tf.global_variable_initializer()
    printGraph()
    tf.session {
      init.run(p to TensorValue(2 x 2, f(1f, 2f, 3f, 4f)))
      a.eval()
    }
  }
  
  @Test
  fun `one hot encoding`() {
    val indices = tf.const(i(0, 2, -1, 1), name = "indices")
    val onehot = tf.one_hot_encoding(indices, 3, 5f, 0f)
    
    println(tf.debugString())
    tf.session {
      val result = eval<Float>(onehot)
      println(result)
    }
  }
}