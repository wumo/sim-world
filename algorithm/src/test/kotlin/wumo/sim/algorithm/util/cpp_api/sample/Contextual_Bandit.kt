package wumo.sim.algorithm.util.cpp_api.sample

import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.bytedeco.javacpp.tensorflow.DT_INT32
import org.junit.Test
import wumo.sim.algorithm.util.cpp_api.BaseTest
import wumo.sim.algorithm.util.cpp_api.contrib.one_hot_encoding
import wumo.sim.algorithm.util.cpp_api.ops.*
import wumo.sim.algorithm.util.cpp_api.tensor
import wumo.sim.algorithm.util.dim
import wumo.sim.util.math.Rand
import java.util.*

class Contextual_Bandit : BaseTest() {
  
  
  @Test
  fun test() {
    //https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c
    var state = 0
    val bandits = arrayOf(floatArrayOf(0.2f, 0f, -0.2f, -5f),
                          floatArrayOf(0.1f, -5f, 1f, 0.25f),
                          floatArrayOf(-5f, 5f, 5f, 5f))
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
    
    println(tf.debugString())
  }
}