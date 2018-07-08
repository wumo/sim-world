package wumo.sim.algorithm.tensorflow.samples

import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.DT_INT32
import org.junit.Test
import wumo.sim.algorithm.tensorflow.contrib.fully_connected
import wumo.sim.algorithm.tensorflow.contrib.one_hot_encoding
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.util.dim
import wumo.sim.algorithm.util.helpers.a
import wumo.sim.algorithm.util.helpers.f
import wumo.sim.algorithm.util.helpers.i
import wumo.sim.util.math.Rand

class Contextual_Bandit : BaseTest() {
  @Test
  fun test() {
    //https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c
    var state = 0
    val bandits = a(f(0.2f, 0f, -0.2f, -5f),
                    f(0.1f, -5f, 1f, 0.25f),
                    f(-5f, 5f, 5f, 5f))
    val num_bandits = bandits.size
    val num_actions = bandits[0].size
    
    fun getbandit(): Int {
      state = Rand().nextInt(0, bandits.size)
      return state
    }
    
    fun pullArm(action: Int): Float {
      val bandit = bandits[state][action]
      return if (Rand().nextGaussian() > bandit) 1f else -1f
    }
    
    val state_in = tf.placeholder(dim(1), dtype = DT_INT32, name = "state_in")
    val state_in_OH = tf.one_hot_encoding(state_in, num_bandits)
    var output = tf.fully_connected(state_in_OH, num_actions,
                                    biases_initializer = null,
                                    activation_fn = { tf.sigmoid(it) },
                                    weights_initializer = tf.ones_initializer())
    output = tf.reshape(output, tf.const(i(-1)), name = "output")
    val chosen_action = tf.argmax(output, 0, name = "chosen_action")
    
    val reward_holder = tf.placeholder(dim(1), dtype = tensorflow.DT_FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(dim(1), dtype = DT_INT32, name = "action_holder")
    
    val responsible_output = tf.slice(output, action_holder, tf.const(i(1)), name = "responsible_weight")
    val loss = tf.neg(tf.mul(tf.log(responsible_output), reward_holder))
    val train = tf.gradientDescentOptimizer(0.001f, loss)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
  }
}
