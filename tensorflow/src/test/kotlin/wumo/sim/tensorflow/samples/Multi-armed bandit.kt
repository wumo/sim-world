package wumo.sim.tensorflow.samples

import org.junit.Test
import wumo.sim.tensorflow.ops.BaseTest
import wumo.sim.tensorflow.ops.basic.times
import wumo.sim.tensorflow.ops.basic.unaryMinus
import wumo.sim.tensorflow.ops.training.GradientDescentOptimizer
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.Rand
import wumo.sim.util.Shape
import wumo.sim.util.f
import wumo.sim.util.i
import wumo.sim.util.ndarray.NDArray
import java.util.*

class `Multi-armed bandit` : BaseTest() {
  @Test
  fun test() {
    val bandit_arms = f(0.2f, 0f, -0.2f, -5f)
    val num_arms = bandit_arms.size
    fun pullBandit(bandit: Float): Float {
      val result = Rand().nextGaussian()
      return if (result > bandit) 1f else -1f
    }
  
    //https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
    val weights = tf.variable(Shape(4), 1f, name = "weights")
    val chosen_action = tf.argmax(weights.toOutput(), 0, name = "chosen_action")
  
    val reward_holder = tf.placeholder(Shape(1), dtype = FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(Shape(1), dtype = INT32, name = "action_holder")
  
    val responsible_weight = tf.slice(weights.toOutput(), action_holder, tf.const(i(1)), name = "responsible_weight")
    val loss = -(tf.log(responsible_weight) * reward_holder)
//    val loss = tf.neg(tf.mul(tf.log(responsible_output), reward_holder))
    val optimizer = GradientDescentOptimizer(learningRate = { 0.001f })
    val train = optimizer.minimize(loss, name = "train")
    val init = tf.globalVariablesInitializer()
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
        feed(reward_holder to NDArray(Shape(1), f(reward)),
             action_holder to NDArray(Shape(1), i(action)))
        train.run()
        total_reward[action] += reward
        if (i % 50 == 0)
          println("Running reward for the $num_arms bandits: ${Arrays.toString(total_reward)}")
        i++
      }
    }
  }
}