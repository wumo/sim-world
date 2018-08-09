package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.gen.mul
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.tensorflow.training.GradientDescentOptimizer

class GradientsKtTest : BaseTest() {
  
  @Test
  fun gradientDescentOptimizer() {
    val x = tf.variable(1f, name = "x")
    val y = tf.variable(1f, name = "y")
    val z = tf.mul(x, y, "z")
    val optimizer = GradientDescentOptimizer(learningRate = 0.1f)
    val opt = optimizer.minimize(z)
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