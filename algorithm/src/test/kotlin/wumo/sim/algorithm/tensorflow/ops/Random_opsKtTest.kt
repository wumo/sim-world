package wumo.sim.algorithm.tensorflow.ops

import org.junit.Test

import org.junit.Assert.*
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.util.x

class Random_opsKtTest : BaseTest() {
  
  @Test
  fun random_uniform() {
    val a = tf.random_uniform(2 x 2)
    val b = tf.random_uniform(2 x 2, 1f, 2f)
    val c = tf.variable(tf.random_uniform(2 x 3, 2f, 3f))
    val init = tf.global_variable_initializer()
    printGraph()
    tf.session {
      init.run()
      a.eval()
      b.eval()
      c.eval()
    }
  }
}