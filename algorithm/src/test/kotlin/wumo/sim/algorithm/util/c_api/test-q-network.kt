package wumo.sim.algorithm.util.c_api

import org.junit.Test
import wumo.sim.algorithm.util.c_api.core.placeholder
import wumo.sim.algorithm.util.c_api.core.variable
import wumo.sim.algorithm.util.c_api.random_ops.random_uniform
import wumo.sim.algorithm.util.x

class `test-q-network` {
  @Test
  fun `q-network`() {
    val tf = TF_C()
    
    val inputs = tf.placeholder(1 x 16, name = "inputs")
    val W = tf.variable(16 x 4, tf.random_uniform(16 x 4), name = "W")
  }
}