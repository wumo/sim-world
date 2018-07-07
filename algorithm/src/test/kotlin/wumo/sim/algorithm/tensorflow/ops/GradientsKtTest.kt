package wumo.sim.algorithm.tensorflow.ops

import org.junit.Test

import org.junit.Assert.*

class GradientsKtTest : BaseTest() {

  @Test
  fun gradientDescentOptimizer() {
    val x = tf.variable(1f, name = "x")
    val y = tf.variable(1f, name = "y")
    val z = tf.mul(x, y, "z")
    val opt = tf.gradientDescentOptimizer(0.1f, z)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      init.run()
      x.eval()
      y.eval()
      z.eval()
      for (i in 0 until 10) {
        println(i)
        opt.run()
        x.eval()
        y.eval()
        z.eval()
      }
    }
  }
}