package wumo.sim.algorithm.drl.deepq.experiments

import org.junit.Test
import wumo.sim.algorithm.drl.deepq.learn
import wumo.sim.algorithm.drl.deepq.mlp
import wumo.sim.envs.classic_control.CartPole

class train_cartpole {
  @Test
  fun train() {
    val env = CartPole()
    val model = mlp(64)
    val act = learn(env,
                       q_func = model,
                       lr = 1e-3f,
                       max_timesteps = 100000,
                       buffer_size = 50000,
                       exploration_fraction = 0.1f,
                       exploration_final_eps = 0.02f,
                       print_freq = 10)
    
  }
}