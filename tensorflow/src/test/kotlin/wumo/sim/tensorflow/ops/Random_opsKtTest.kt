package wumo.sim.tensorflow.ops

import org.junit.Test
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.i
import wumo.sim.util.x

class Random_opsKtTest : BaseTest() {
  
  @Test
  fun random_uniform() {
    val a = tf.randomUniform(tf.const(i(2, 2)), FLOAT)
    val b = tf.random_uniform(2 x 2, 1f, 2f)
    val c = tf.variable(tf.random_uniform(2 x 3, 2f, 3f))
    val init = tf.globalVariablesInitializer()
    printGraph()
    tf.session {
      init.run()
      a.eval()
      b.eval()
      c.eval()
    }
  }
  
  @Test
  fun random_uniform_int() {
    val a = tf.random_uniform(tf.const(i(1)), min = 0, max = 2, dtype = INT32)
    tf.session {
      repeat(10) {
        a.eval()
      }
    }
  }
}