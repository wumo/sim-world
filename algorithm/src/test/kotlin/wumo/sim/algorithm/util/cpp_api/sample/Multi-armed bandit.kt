package wumo.sim.algorithm.util.cpp_api.sample

import org.bytedeco.javacpp.tensorflow
import org.junit.Test
import wumo.sim.algorithm.util.cpp_api.BaseTest
import wumo.sim.algorithm.util.cpp_api.ops.*
import wumo.sim.algorithm.util.dim

class `Multi-armed bandit` : BaseTest() {
  @Test
  fun test() {
    //https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
    val weights = tf.variable(dim(4), 1f, name = "weights")
    val chosen_action = tf.argmax(weights, 0, name = "chosen_action")
    
    val reward_holder = tf.placeholder(dim(1), dtype = tensorflow.DT_FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(dim(1), dtype = tensorflow.DT_INT32, name = "action_holder")
    val responsible_weight = tf.slice(weights, action_holder, tf.const(intArrayOf(1)), name = "responsible_weight")
    val loss = tf.neg(tf.mul(tf.log(responsible_weight), reward_holder))
    val train = tf.gradientDescentOptimizer(0.001f, loss)
    val init = tf.global_variable_initializer()
    println(tf.debugString())
  }
}