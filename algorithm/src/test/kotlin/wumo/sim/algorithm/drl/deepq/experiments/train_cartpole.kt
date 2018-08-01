package wumo.sim.algorithm.drl.deepq.experiments

import org.junit.Test
import wumo.sim.algorithm.drl.deepq.learn
import wumo.sim.algorithm.drl.deepq.loadModel
import wumo.sim.algorithm.drl.deepq.mlp
import wumo.sim.algorithm.tensorflow.defaut
import wumo.sim.envs.classic_control.CartPole
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.newaxis
import wumo.sim.wrappers.TimeLimit

class train_cartpole {
  @Test
  fun train() {
    val env = TimeLimit(CartPole(), 200)
    val model = mlp(64)
    learn(env,
          q_func = model,
          model_file_path = "cartpole.model",
          lr = 1e-3f,
          max_timesteps = 100000,
          buffer_size = 50000,
          exploration_fraction = 0.1f,
          exploration_final_eps = 0.02f,
          print_freq = 10)
  }
  
  @Test
  fun enjoy() {
    val env = CartPole()
    val (tf, init, act) = loadModel("cartpole.model")
    defaut(tf) {
      tf.session {
        init.run()
        while (true) {
          var _obs = env.reset()
          var _done = false
          var episode_rew = 0f
          var interval = 200
          while (!_done) {
            env.render()
            interval--
            val action = if (interval < 0) {
              if (interval < -4)
                interval = 200
              0
            } else
              act(newaxis(NDArray.toNDArray(_obs)), stochastic = false)[0].get() as Int
            val (obs, rew, done) = env.step(action)
            _done = done
            _obs = obs
            episode_rew += rew
          }
          println("Episode reward $episode_rew")
        }
      }
    }
  }
}