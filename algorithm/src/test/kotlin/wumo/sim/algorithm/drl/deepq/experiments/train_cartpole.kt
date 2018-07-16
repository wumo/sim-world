package wumo.sim.algorithm.drl.deepq.experiments

import org.junit.Test
import wumo.sim.algorithm.drl.deepq.learn
import wumo.sim.algorithm.drl.deepq.mlp
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.envs.classic_control.CartPole

class train_cartpole {
  @Test
  fun train() {
    val env = CartPole()
    val model = tf.mlp(64)
    val act = tf.learn(env,
                       q_func = model,
                       lr = 1e-3,
                       max_timesteps = 100000,
                       buffer_size = 50000,
                       exploration_fraction = 0.1,
                       exploration_final_eps = 0.02,
                       print_freq = 10)
    
  }
}