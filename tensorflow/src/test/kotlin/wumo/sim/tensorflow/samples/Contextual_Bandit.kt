package wumo.sim.tensorflow.samples

import org.junit.Test
import wumo.sim.tensorflow.contrib.layers
import wumo.sim.tensorflow.ops.BaseTest
import wumo.sim.tensorflow.ops.basic.times
import wumo.sim.tensorflow.ops.basic.unaryMinus
import wumo.sim.tensorflow.ops.training.GradientDescentOptimizer
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray

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
  
    val state_in = tf.placeholder(Shape(1), dtype = INT32, name = "state_in")
    val state_in_OH = layers.one_hot_encoding(state_in, num_bandits)
    var output = layers.fully_connected(state_in_OH, num_actions,
                                        biases_initializer = null,
                                        activation_fn = { tf.sigmoid(it) },
                                        weights_initializer = tf.onesInitializer())
    output = tf.reshape(output, tf.const(i(-1)), name = "output")
    val chosen_action = tf.argmax(output, 0, name = "chosen_action")
  
    val reward_holder = tf.placeholder(Shape(1), dtype = FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(Shape(1), dtype = INT32, name = "action_holder")
  
    val responsible_output = tf.slice(output, action_holder, tf.const(i(1)), name = "responsible_weight")
    val loss = -(tf.log(responsible_output) * reward_holder)
//    val loss = tf.neg(tf.mul(tf.log(responsible_output), reward_holder))
    val optimizer = GradientDescentOptimizer(learningRate = { 0.001f })
    val train = optimizer.minimize(loss, name = "train")
//    val train = tf.gradientDescentOptimizer(0.001f, loss)
    val init = tf.globalVariablesInitializer()
    printGraph()
    val weights = tf.currentGraph.trainableVariables.first()
    val total_episodes = 10000
    val total_reward = NDArray.zeros(Shape(num_bandits))
    val e = 0.1
    tf.session {
      init.run()
      var i = 0
      var ww: NDArray<Float>? = null
      while (i < total_episodes) {
        val s = getbandit()
        val action = if (Rand().nextDouble() < e)
          Rand().nextInt(num_actions)
        else {
          feed(state_in to NDArray(Shape(1), i(s)))
          eval<Int>(chosen_action).get()
        }
        val reward = pullArm(action)
      
        feed(reward_holder to NDArray(Shape(1), f(reward)),
             action_holder to NDArray(Shape(1), i(action)),
             state_in to NDArray(Shape(1), i(s)))
        target(train)
        ww = eval(weights)
      
        total_reward[s] += reward
        if (i % 500 == 0) {
          print("Mean reward for each of the $num_bandits bandits: [")
          for (i in 0 until num_bandits) {
            print("${total_reward[i] / num_actions},")
          }
          println("]")
        }
        i++
      }
      ww!!
      for (i in 0 until num_bandits) {
        val best_a = argmax(0 until num_actions) { ww[i, it] }
        val actua_a = argmin(0..bandits[i].lastIndex) { bandits[i][it] }
        println("The agent thinks action $best_a for bandit $i is the most promising..." +
                    "and it was ${if (best_a == actua_a) "right" else "wrong"}")
      }
    }
  }
  
  
}
