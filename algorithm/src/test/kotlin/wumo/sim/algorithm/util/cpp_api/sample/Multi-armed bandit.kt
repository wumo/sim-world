package wumo.sim.algorithm.util.cpp_api.sample

import org.bytedeco.javacpp.tensorflow
import org.junit.Test
import wumo.sim.algorithm.util.cpp_api.BaseTest
import wumo.sim.algorithm.util.cpp_api.ops.*
import wumo.sim.algorithm.util.cpp_api.tensor
import wumo.sim.algorithm.util.dim
import wumo.sim.util.math.Rand
import java.util.*

class `Multi-armed bandit` : BaseTest() {
  @Test
  fun test() {
    val bandit_arms = floatArrayOf(0.2f, 0f, -0.2f, -5f)
    val num_arms = bandit_arms.size
    fun pullBandit(bandit: Float): Float {
      val result = Rand().nextGaussian()
      return if (result > bandit) 1f else -1f
    }
    
    //https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
    val weights = tf.variable(dim(4), 1f, name = "weights")
    val chosen_action = tf.argmax(weights, 0, name = "chosen_action")
    
    val reward_holder = tf.placeholder(dim(1), dtype = tensorflow.DT_FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(dim(1), dtype = tensorflow.DT_INT32, name = "action_holder")
    
    val responsible_output = tf.slice(weights, action_holder, tf.const(intArrayOf(1)), name = "responsible_weight")
    val loss = tf.neg(tf.mul(tf.log(responsible_output), reward_holder))
    val train = tf.gradientDescentOptimizer(0.001f, loss)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    
    val total_episodes = 1000//Set total number of episodes to train agent on.
    val total_reward = FloatArray(num_arms) { 0f }
    val e = 0.1
    tf.session {
      init.run()
      var i = 0
      while (i < total_episodes) {
        val action = if (Rand().nextDouble() < e)
          Rand().nextInt(num_arms)
        else
          eval<Int>(chosen_action).get()
        val reward = pullBandit(bandit_arms[action])//Get our reward from picking one of the bandits.
        feed(reward_holder to tensor(reward),
             action_holder to tensor(action))
        train.run()
        total_reward[action] += reward
        if (i % 50 == 0)
          println("Running reward for the $num_arms bandits: ${Arrays.toString(total_reward)}")
        i++
      }
    }
  }
}