package wumo.sim.tensorflow.samples

import org.junit.Test
import wumo.sim.envs.classic_control.CartPole
import wumo.sim.tensorflow.contrib.layers
import wumo.sim.tensorflow.ops.*
import wumo.sim.tensorflow.ops.training.AdamOptimizer
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.INT32
import wumo.sim.util.Shape
import wumo.sim.util.i

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
    val state_in = tf.placeholder(Shape(-1, s_size), FLOAT, name = "state_in")
    val hidden = layers.fully_connected(state_in, h_size,
                                             biases_initializer = null,
                                             activation_fn = { tf.relu(it) })
    val output = layers.fully_connected(hidden, a_size,
                                    biases_initializer = null,
                                    activation_fn = { tf.softmax(it) })
    val chosen_action = tf.argmax(output, 1, name = "chosen_action")
    
    //The next six lines establish the training proceedure. We feed the reward and chosen action into the network
    //to compute the loss, and use it to update the network.
    val reward_holder = tf.placeholder(Shape(-1), FLOAT, name = "reward_holder")
    val action_holder = tf.placeholder(Shape(-1), INT32, name = "action_holder")
    
    val indexes = tf.range(tf.const(0), tf.shape(output)[0]) * tf.shape(output)[1] + action_holder
    val responsible_output = tf.gather(tf.reshape(output, tf.const(i(-1))), indexes, name = "responsible_weight")
    
    val loss = -tf.mean(tf.log(responsible_output) * reward_holder)
    
    val gradient_holders = mutableListOf<Output>()
    for ((idx, v) in tf.currentGraph.trainableVariables.withIndex())
      gradient_holders += tf.placeholder(name = "${idx}_holder")
    
    val gradients = tf.gradients(listOf(loss), tf.currentGraph.trainableVariables.map {
      it.toOutput()
    })
    val optimizer = AdamOptimizer(learningRate = { lr })
    val update_batch = optimizer.applyGradients(gradients.zip(tf.currentGraph.trainableVariables))
    val init = tf.globalVariablesInitializer()
    printGraph()
    
    val total_episodes = 5000  //Set total number of episodes to train agent on.
    val max_ep = 999
    val update_frequency = 5
    tf.session {
      init.run()
      var i = 0
      
      val gradBuffer = eval(tf.currentGraph.trainableVariables)
//      for ((ix, grad) in gradBuffer.withIndex()) {
//        gradBuffer[ix] = 0
//      }
      while (i < total_episodes) {
        val s = env.reset()
        var running_reward = 0f
        for (j in 0 until max_ep) {
//          feed(state_in to Tensor(f(s)))
//          output.eval()
        }
      }
    }
  }
}