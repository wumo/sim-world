package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.tensorflow.ops.training.AdamOptimizer
import wumo.sim.tensorflow.ops.training.GradientDescentOptimizer
import wumo.sim.tensorflow.tf

class GradientsOldKtTest : BaseTest() {
  
  @Test
  fun gradients() {
    val x = tf.variable(1f, name = "x")
    val y = x * x
    val trainer = GradientDescentOptimizer({ 0.1f })
    val update = trainer.minimize(y)
    val init = tf.globalVariablesInitializer()
    printGraph()
    tf.session {
      init.run()
      repeat(20) {
        update.run()
        val (_x, _y) = eval<Float, Float>(x.toOutput(), y)
        println("$_y,$_x")
      }
    }
  }
  
  @Test
  fun gradientDescentOptimizer() {
    val x = tf.variable(1f, name = "x")
    val y = tf.variable(1f, name = "y")
    val z = tf.mul(x.toOutput(), y.toOutput(), "z")
    val optimizer = AdamOptimizer(learningRate = { 0.1f })
    val opt = optimizer.minimize(z)
    val init = tf.globalVariablesInitializer()
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