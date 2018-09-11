package wumo.sim.algorithm.drl.deepq.experiments

import org.junit.Test
import wumo.sim.algorithm.drl.common.identity
import wumo.sim.algorithm.drl.deepq.learn
import wumo.sim.envs.classic_control.CartPole
import wumo.sim.wrappers.TimeLimit

class train_cartpole {
  @Test
  fun train() {
    val env = TimeLimit(CartPole(), 200)
    learn(env,
          network = identity,
          learning_rate = 1e-3f,
          total_timesteps = 10_0000,
          buffer_size = 5_0000,
          exploration_fraction = 0.1f,
          exploration_final_eps = 0.02f,
          print_freq = 10,
          hiddens = listOf(64),
          dueling = false,
          layer_norm = false)
    
  }
}