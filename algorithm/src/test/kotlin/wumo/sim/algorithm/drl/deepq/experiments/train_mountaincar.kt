package wumo.sim.algorithm.drl.deepq.experiments

import org.junit.Test
import wumo.sim.algorithm.drl.deepq.learn
import wumo.sim.algorithm.drl.deepq.loadModel
import wumo.sim.algorithm.drl.deepq.mlp
import wumo.sim.tensorflow.defaut
import wumo.sim.envs.classic_control.MountainCar
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.newaxis
import wumo.sim.wrappers.TimeLimit

class train_mountaincar {
  @Test
  fun train() {
    val env = TimeLimit(MountainCar(), max_episode_steps = 200)
//    val env = MountainCar()
    val model = mlp(64, layer_norm = true)
    learn(env,
          q_func = model,
          model_file_path = "MountainCar.model",
          lr = 1e-3f,
          max_timesteps = 1000000,
          buffer_size = 50000,
          exploration_fraction = 0.1f,
          exploration_final_eps = 0.1f,
          print_freq = 10,
          param_noise = true)
  }
  
  @Test
  fun enjoy() {
    val env = TimeLimit(MountainCar(), max_episode_steps = 200)
    val (tf, init, act) = loadModel("MountainCar.model")
    defaut(tf) {
      tf.session {
        init.run()
        while (true) {
          var _obs = env.reset()
          var _done = false
          var episode_rew = 0f
          while (!_done) {
            env.render()
            val action = act(newaxis(NDArray.toNDArray(_obs)), stochastic = false)[0].get() as Int
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