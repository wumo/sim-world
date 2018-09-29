package wumo.sim.algorithm.drl.deepq.experiments

import org.junit.Test
import wumo.sim.algorithm.drl.common.identity
import wumo.sim.algorithm.drl.deepq.learn
import wumo.sim.algorithm.drl.deepq.loadModel
import wumo.sim.envs.envs
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.newaxis

class train_mountaincar {
  @Test
  fun train() {
    val env = envs.`MountainCar-v0`()
    learn("mountaincar.model",
          env = env,
          network = identity,
          learning_rate = 1e-3f,
          total_timesteps = 10_0000,
          buffer_size = 5_0000,
          exploration_fraction = 0.1f,
          exploration_final_eps = 0.1f,
          print_freq = 10,
          param_noise = true,
          hiddens = listOf(64),
          dueling = false,
          layer_norm = true)
  }
  
  @Test
  fun enjoy() {
    val env = envs.`MountainCar-v0`()
    val (graph, init, act) = loadModel("mountaincar.model")
    tf.unsafeDefaultGraph(graph) {
      tf.session {
        init.run()
        while (true) {
          var _obs = env.reset()
          var _done = false
          var episode_rew = 0f
          while (!_done) {
            env.render()
            val action = act(newaxis(NDArray.toNDArray(_obs)),
                             stochastic = false)[0][0] as Long
            val (obs, rew, done) = env.step(action.toInt())
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