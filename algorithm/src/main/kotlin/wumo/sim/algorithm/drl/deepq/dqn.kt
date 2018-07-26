package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.algorithm.tensorflow.training.AdamOptimizer
import wumo.sim.core.Env
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.abs
import wumo.sim.util.ndarray.ones_like
import wumo.sim.util.ndarray.plus

/**
 * Train a deepq model.
 *
 * Parameters
 * @param env: gym.Env environment to train on
 * @param q_func: (tf.Variable, int, str, bool) -> tf.Variable the model that takes the following inputs:
 * @param observation_in: the output of observation placeholder
 * @param num_actions: number of actions
 * @param lr: learning rate for adam optimizer
 * @param max_timesteps: number of env steps to optimizer for
 * @param buffer_size: size of the replay buffer
 * @param exploration_fraction: fraction of entire training period over which the exploration rate is annealed
 * @param exploration_final_eps: final value of random action probability
 * @param train_freq: update the model every `train_freq` steps. set to None to disable printing
 * @param batch_size: size of a batched sampled from replay buffer for training
 * @param print_freq: how often to print out training progress set to None to disable printing
 * @param learning_starts: how many steps of the model to collect transitions for before learning starts
 * @param gamma: discount factor
 * @param target_network_update_freq: update the target network every `target_network_update_freq` steps.
 * @param prioritized_replay: if True prioritized replay buffer will be used.
 * @param prioritized_replay_alpha: alpha parameter for prioritized replay buffer
 * @param prioritized_replay_beta0: initial value of beta for prioritized replay buffer
 * @param prioritized_replay_beta_iters: number of iterations over which beta will be annealed
 * from initial value to 1.0. If set to None equals to max_timesteps.
 * @param prioritized_replay_eps: epsilon to add to the TD errors when updating priorities.
 * @return act: Wrapper over act function. Adds ability to save it and load it.
 */
fun <O : Any, A : Any> learn(env: Env<O, A>,
                             q_func: Q_func,
                             lr: Float = 5e-4f,
                             max_timesteps: Int = 100000,
                             buffer_size: Int = 50000,
                             exploration_fraction: Float = 0.1f,
                             exploration_final_eps: Float = 0.02f,
                             train_freq: Int = 1,
                             batch_size: Int = 32,
                             print_freq: Int = 100,
                             learning_starts: Int = 1000,
                             gamma: Float = 1.0f,
                             target_network_update_freq: Int = 500,
                             prioritized_replay: Boolean = false,
                             prioritized_replay_alpha: Float = 0.6f,
                             prioritized_replay_beta0: Float = 0.4f,
                             prioritized_replay_beta_iters: Int? = null,
                             prioritized_replay_eps: Float = 1e-6f,
                             param_noise: Boolean = false) {
  fun make_obs_ph(name: String) = ObservationInput(env.observation_space, name = name)
  
  val (act, train, update_target, debug) = build_train(
      make_obs_ph = ::make_obs_ph,
      q_func = q_func,
      num_actions = env.action_space.n,
      optimizer = AdamOptimizer(learningRate = lr),
      gamma = gamma,
      grad_norm_clipping = tf.const(10),
      param_noise = param_noise)
  
  //Create the replay buffer
  val replay_buffer: ReplayBuffer<O, A>
  val beta_schedule: Schedule
  if (prioritized_replay) {
    replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha = prioritized_replay_alpha)
    beta_schedule = LinearSchedule(schedule_timesteps = prioritized_replay_beta_iters ?: max_timesteps,
                                   initial_p = prioritized_replay_beta0,
                                   final_p = 1f)
  } else {
    replay_buffer = ReplayBuffer(buffer_size)
    beta_schedule = NoneSchedule()
  }
  
  //Create the schedule for exploration starting from 1.
  val exploration = LinearSchedule(schedule_timesteps = (exploration_fraction * max_timesteps).toInt(),
                                   initial_p = 1f,
                                   final_p = exploration_final_eps)
  
  //Initialize the parameters and copy them to the target network.
  tf.session {
    val init = tf.global_variable_initializer()
    init.run()
    update_target()
    
    val episode_rewards = mutableListOf(0f)
    var saved_mean_reward = 0f
    var obs = env.reset()
    var reset = true
    
    for (t in 0 until max_timesteps) {
      //Take action and update exploration to the newest value
      var update_eps: Float
      var update_param_noise_threshold: Float
      val action = if (!param_noise) {
        update_eps = exploration.value(t)
        act(NDArray.toNDArray(obs), update_eps = update_eps)
      } else {
        update_eps = 0f
        // Compute the threshold such that the KL divergence between perturbed and non-perturbed
        // policy is comparable to eps-greedy exploration with eps = exploration.value(t).
        // See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
        // for detailed explanation.
        update_param_noise_threshold = (-Math.log((1 - exploration.value(t) + exploration.value(t) / env.action_space.n).toDouble())).toFloat()
        act as ActWithParamNoise
        act(NDArray.toNDArray(obs), reset, update_param_noise_threshold, update_param_noise_scale = true, update_eps = update_eps)
      }[0].get() as A
      val env_action = action
      reset = false
      val (new_obs, rew, done, _) = env.step(env_action as A)
      //Store transition in the replay buffer.
      replay_buffer.add(obs, action, rew, new_obs, done)
      obs = new_obs
      
      episode_rewards[episode_rewards.lastIndex] += rew.toFloat()
      if (done) {
        obs = env.reset()
        episode_rewards += 0f
        reset = true
      }
      
      if (t > learning_starts && t % train_freq == 0) {
        //Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if (prioritized_replay) {
          replay_buffer as PrioritizedReplayBuffer
          val (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) =
              replay_buffer.sample(batch_size, beta = beta_schedule.value(t))
          val (td_errors) = train(obses_t, actions, rewards, obses_tp1, dones, weights)
          
          val new_priorities = abs(td_errors) + prioritized_replay_eps
          replay_buffer.update_priorities(batch_idxes, new_priorities)
        } else {
          val (obses_t, actions, rewards, obses_tp1, dones) = replay_buffer.sample(batch_size)
          val weights = ones_like(rewards)
          train(obses_t, actions, rewards, obses_tp1, dones, weights)
        }
      }
      
      if (t > learning_starts && t % target_network_update_freq == 0)
        update_target()
      
      val mean_100ep_reward = episode_rewards.mean()
      val num_episodes = episode_rewards.size
      if (done && episode_rewards.size % print_freq == 0) {
        println("steps:t\n" +
                "episodes: $num_episodes\n" +
                "mean 100 episode reward: $mean_100ep_reward\n" +
                "${100 * exploration.value(t)} time spent exploring")
      }
      
      
    }
  }
}

private fun MutableList<Float>.mean() = sum() / size
