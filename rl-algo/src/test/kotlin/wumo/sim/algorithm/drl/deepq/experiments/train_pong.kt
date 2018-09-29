package wumo.sim.algorithm.drl.deepq.experiments

import org.bytedeco.javacpp.Pointer
import org.junit.Test
import wumo.sim.algorithm.drl.common.convOnly
import wumo.sim.algorithm.drl.common.make_atari
import wumo.sim.algorithm.drl.common.wrap_atari_dqn
import wumo.sim.algorithm.drl.deepq.learn
import wumo.sim.algorithm.drl.deepq.loadModel
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.newaxis
import wumo.sim.util.t3

class train_pong {
  @Test
  fun train() {
//    System.setProperty("org.bytedeco.javacpp.logger.debug", "true")
    Pointer.maxPhysicalBytes()
    val _env = make_atari("PongNoFrameskip-v4")
    val env = wrap_atari_dqn(_env)
    learn("PongNoFrameskip-v4.model",
          env = env,
          network = convOnly(convs = listOf(t3(32, 8, 4),
                                            t3(64, 4, 2),
                                            t3(64, 3, 1))),
          learning_rate = 1e-4f,
          total_timesteps = 1000_0000,
          buffer_size = 1000,
          exploration_fraction = 0.1f,
          exploration_final_eps = 0.01f,
          train_freq = 4,
          learning_starts = 1000,
          gamma = 0.99f,
          print_freq = 10,
          hiddens = listOf(256),
          dueling = true,
          layer_norm = false)
  }
  
  @Test
  fun enjoy() {
    val env = make_atari("PongNoFrameskip-v4")
    val (graph, init, act) = loadModel("PongNoFrameskip-v4.model")
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
                             stochastic = false)[0].get() as Long
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