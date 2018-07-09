package wumo.sim.algorithm.tensorflow.samples

import org.bytedeco.javacpp.tensorflow.*
import org.junit.Test
import wumo.sim.algorithm.tensorflow.contrib.fully_connected
import wumo.sim.algorithm.tensorflow.contrib.one_hot_encoding
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.util.dim
import wumo.sim.algorithm.util.helpers.i
import wumo.sim.algorithm.util.x
import wumo.sim.envs.classic_control.CartPole

class VanillaPolicyTest : BaseTest() {
  @Test
  fun test() {
    val env = CartPole()
    val gamma = 0.99
    fun discount_reward(r: Float) {
    
    }
    
    val s_size = 4
    val a_size = 2
    val h_size = 8
    val state_in = tf.placeholder(-1 x s_size, dtype = DT_FLOAT, name = "state_in")
    val hidden = tf.fully_connected(state_in, h_size,
                                    biases_initializer = null,
                                    activation_fn = { tf.relu(it) })
    val output = tf.fully_connected(hidden, a_size,
                                    biases_initializer = null,
                                    activation_fn = { tf.softmax(it) })
    val chosen_action = tf.argmax(output, 1, name = "chosen_action")
    
    val reward_holder = tf.placeholder(dim(-1), dtype = DT_FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(dim(-1), dtype = DT_INT32, name = "action_holder")
    
    
    val responsible_output = tf.slice(output, action_holder, tf.const(i(1)), name = "responsible_weight")
    val loss = tf.neg(tf.mul(tf.log(responsible_output), reward_holder))
    val train = tf.gradientDescentOptimizer(0.001f, loss)
    val init = tf.global_variable_initializer()
    printGraph()
  }
}