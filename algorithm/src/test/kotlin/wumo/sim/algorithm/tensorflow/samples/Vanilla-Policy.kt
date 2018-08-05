package wumo.sim.algorithm.tensorflow.samples

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.bytedeco.javacpp.tensorflow.DT_INT32
import org.junit.Test
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.contrib.fully_connected
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.ops.gen.log
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.tensorflow.training.AdamOptimizer
import wumo.sim.envs.classic_control.CartPole
import wumo.sim.util.dim
import wumo.sim.util.i
import wumo.sim.util.x
import wumo.sim.util.zip

class VanillaPolicyTest : BaseTest() {
  @Test
  fun test() {
    val env = CartPole()
    val gamma = 0.99
    fun discount_reward(r: Float) {
    }
    
    val lr = 1e-2f
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
    
    //The next six lines establish the training proceedure. We feed the reward and chosen action into the network
    //to compute the loss, and use it to update the network.
    val reward_holder = tf.placeholder(dim(-1), dtype = DT_FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(dim(-1), dtype = DT_INT32, name = "action_holder")
    
    val indexes = tf.range(tf.const(0), tf.shape(output)[0]) * tf.shape(output)[1] + action_holder
    val responsible_output = tf.gather(tf.reshape(output, tf.const(i(-1))), indexes, name = "responsible_weight")
    
    val loss = -tf.mean(tf.log(responsible_output) * reward_holder)
    
    val gradient_holders = mutableListOf<Tensor>()
    for ((idx, v) in tf.trainables.withIndex())
      gradient_holders += tf.placeholder(name = "${idx}_holder")
    
    val gradients = tf.gradients(loss, tf.trainables)
    val optimizer = AdamOptimizer(learningRate = lr)
    val update_batch = optimizer.apply_gradients(gradients.zip(tf.trainables))
    val init = tf.global_variable_initializer()
    printGraph()
    
    val total_episodes = 5000  //Set total number of episodes to train agent on.
    val max_ep = 999
    val update_frequency = 5
    tf.session {
      init.run()
      var i = 0
      
      val gradBuffer = eval(tf.trainables)
//      for ((ix, grad) in gradBuffer.withIndex()) {
//        gradBuffer[ix] = 0
//      }
      while (i < total_episodes) {
        val s = env.reset()
        var running_reward = 0f
        for (j in 0 until max_ep) {
//          feed(state_in to TensorBuffer(f(s)))
//          output.eval()
        }
      }
    }
  }
}