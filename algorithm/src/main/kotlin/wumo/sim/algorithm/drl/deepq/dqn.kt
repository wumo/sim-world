package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.core.Env

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
fun <O, A> TF.learn(env: Env<O, A>,
                    q_func: Q_func,
                    lr: Double = 5e-4,
                    max_timesteps: Int = 100000,
                    buffer_size: Int = 50000,
                    exploration_fraction: Double = 0.1,
                    exploration_final_eps: Double = 0.02,
                    train_freq: Int = 1,
                    batch_size: Int = 32,
                    print_freq: Int = 100,
                    learning_starts: Int = 1000,
                    gamma: Double = 1.0,
                    target_network_update_freq: Int = 500,
                    prioritized_replay: Boolean = false,
                    prioritized_replay_alpha: Double = 0.6,
                    prioritized_replay_beta0: Double = 0.4,
                    prioritized_replay_beta_iters: Any? = null,
                    prioritized_replay_eps: Double = 1e-6,
                    param_noise: Boolean = false) {
  
  
}