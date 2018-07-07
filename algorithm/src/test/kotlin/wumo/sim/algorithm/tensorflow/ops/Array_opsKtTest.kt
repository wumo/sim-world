package wumo.sim.algorithm.tensorflow.ops

import org.junit.Test

import org.junit.Assert.*
import wumo.sim.algorithm.tensorflow.TensorValue
import wumo.sim.algorithm.util.helpers.f
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
}