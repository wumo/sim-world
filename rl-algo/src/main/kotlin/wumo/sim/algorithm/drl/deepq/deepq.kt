package wumo.sim.algorithm.drl.deepq

import org.bytedeco.javacpp.Pointer
import wumo.sim.algorithm.drl.common.LinearSchedule
import wumo.sim.algorithm.drl.common.Schedule
import wumo.sim.core.Env
import wumo.sim.tensorflow.core.TensorFunction
import wumo.sim.tensorflow.ops.training.AdamOptimizer
import wumo.sim.tensorflow.ops.variables.ReuseOrCreateNew
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf
import wumo.sim.util.TimeMeter
import wumo.sim.util.native
import wumo.sim.util.ndarray.*
import wumo.sim.util.ndarray.types.NDInt
import wumo.sim.util.ref
import java.text.NumberFormat

val formatter = NumberFormat.getInstance()
val runtime = Runtime.getRuntime()

fun <O : Any, OE : Any, A : Any, AE : Any> learn(
    model_file_path: String,
    env: Env<O, OE, A, AE, *>,
    network: TensorFunction,
    seed: Int? = null,
    learning_rate: Float = 5e-4f,
    total_timesteps: Int = 10_0000,
    buffer_size: Int = 5_0000,
    exploration_fraction: Float = 0.1f,
    exploration_final_eps: Float = 0.02f,
    train_freq: Int = 1,
    batch_size: Int = 32,
    print_freq: Int = 100,
    checkpoint_freq: Int = 1_0000,
    checkpoint_patch: String? = null,
    learning_starts: Int = 1000,
    gamma: Float = 1f,
    target_network_update_freq: Int = 500,
    prioritized_replay: Boolean = false,
    prioritized_replay_alpah: Float = 0.6f,
    prioritized_replay_beta0: Float = 0.4f,
    prioritized_replay_beta_iters: Int? = null,
    prioritized_replay_eps: Float = 1e-6f,
    param_noise: Boolean = false,
    callback: Any? = null,
    load_path: String? = null,
    hiddens: List<Int> = listOf(256),
    dueling: Boolean = true,
    layer_norm: Boolean = false) {
  
  val q_func = build_q_func(network, hiddens, dueling, layer_norm)
  
  fun makeObservationPlaceholder(name: String) =
      ObservationInput(observation_space = env.observation_space, name = name)
  
  val (act, train, update_target, debug) = build_train(
      makeObsPh = ::makeObservationPlaceholder,
      qFunc = q_func,
      numActions = env.action_space.n,
      optimizer = AdamOptimizer(learningRate = { learning_rate }),
      gamma = gamma,
      gradNormClipping = 10,
      paramNoise = param_noise)
  
  val act_vars = debug["act_vars"] as Set<Variable>
  
  //Create the replay buffer
  val replay_buffer: ReplayBuffer<O, OE, A, AE>
  lateinit var beta_schedule: Schedule
  if (prioritized_replay) {
    replay_buffer = PrioritizedReplayBuffer(buffer_size,
                                            env.observation_space,
                                            env.action_space,
                                            alpha = prioritized_replay_alpah)
    beta_schedule = LinearSchedule(
        schedule_timesteps = prioritized_replay_beta_iters ?: total_timesteps,
        initial_p = prioritized_replay_beta0,
        final_p = 1f)
  } else
    replay_buffer = ReplayBuffer(buffer_size, env.observation_space, env.action_space)
  
  //Create the schedule for exploration starting from 1.
  val exploration = LinearSchedule(
      schedule_timesteps = (exploration_fraction * total_timesteps).toInt(),
      initial_p = 1f,
      final_p = exploration_final_eps)
  
  //Initialize the parameters and copy them to the target network.
  tf.session {
    val init = tf.globalVariablesInitializer()
    init.run()
    update_target()
    
    val episode_rewards = mutableListOf(0f)
    var saved_mean_reward = Float.NEGATIVE_INFINITY
    var obs = env.reset()
    var reset = true
    val meter = TimeMeter()
    for (t in 0..total_timesteps) {
      //Take action and update exploration to the newest value
      var update_eps: Float
      var update_param_noise_threshold: Float
      meter.start("total")
      meter.start("act")
      native {
        val act_result = if (!param_noise) {
          update_eps = exploration.value(t)
          act(newaxis(NDArray.toNDArray<OE>(obs)), update_eps = update_eps)
        } else {
          update_eps = 0f
          // Compute the threshold such that the KL divergence between perturbed and non-perturbed
          // policy is comparable to eps-greedy exploration with eps = exploration.value(t).
          // See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
          // for detailed explanation.
          update_param_noise_threshold = (-Math.log(
              (1 - exploration.value(t) + exploration.value(t) / env.action_space.n).toDouble())).toFloat()
          act as ActWithParamNoise
          act(newaxis(NDArray.toNDArray<OE>(obs)), reset, update_param_noise_threshold, update_param_noise_scale = true, update_eps = update_eps)
        }
        meter.end("act")
        val action = env.action_space.dtype.cast(act_result[0][0]) as A
        val env_action = action
        reset = false
        meter.start("step")
        val (new_obs, rew, done, _) = env.step(env_action)
        meter.end("step")
//      println((end-start)/1e9)
        //Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, done)
        obs = new_obs
        
        episode_rewards[episode_rewards.lastIndex] += rew
        if (done) {
          obs = env.reset()
          (obs as? NDArray<*>)?.ref()
          episode_rewards += 0f
          reset = true
        }
        
        if (t > learning_starts && t % train_freq == 0) {
          //Minimize the error in Bellman's equation on a batch sampled from replay buffer.
          if (prioritized_replay) {
            replay_buffer as PrioritizedReplayBuffer<*, *, *, *>
            val (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) =
                replay_buffer.sample(batch_size, beta = beta_schedule.value(t))
            val (td_errors) = train(obses_t, actions, rewards, obses_tp1, dones, weights)
            
            val new_priorities = abs(td_errors) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)
          } else {
            val (obses_t, actions, rewards, obses_tp1, dones) = replay_buffer.sample(batch_size)
//          actions as NDArray<Long>
//          val _actions = actions.cast(NDInt)
            val weights = ones_like(rewards)
            train(obses_t, actions, rewards, obses_tp1, dones, weights)
          }
        }
        
        if (t > learning_starts && t % target_network_update_freq == 0)
          update_target()
        
        val mean_100ep_reward = episode_rewards.mean(-101, -1)
        val num_episodes = episode_rewards.size
        if (done && episode_rewards.size % print_freq == 0) {
          println("steps:$t\n" +
                      "episodes: $num_episodes\n" +
                      "mean 100 episode reward: $mean_100ep_reward\n" +
                      "${100 * exploration.value(t)} time spent exploring")
        }
        if (checkpoint_freq > 0 && t > learning_starts && num_episodes > 100 && t % checkpoint_freq == 0) {
          if (mean_100ep_reward > saved_mean_reward) {
            if (print_freq > 0)
              System.err.println("Saving model due to mean reward increase: $saved_mean_reward -> $mean_100ep_reward")
            saved_mean_reward = mean_100ep_reward
            val result = eval(act_vars)
            saveVariable(act_vars.map { it.name }.zip(result))
          }
        }
        meter.end("total")
      }
      if (t % 300 == 0) {
        println("$t: ${replay_buffer.size}:" +
                    "phy=" + formatter.format(Pointer.physicalBytes()) + ", " +
                    "max=" + formatter.format(Pointer.maxPhysicalBytes()) + "," +
                    "heap=" + formatter.format(runtime.totalMemory()) + "," +
                    "used=" + formatter.format(runtime.totalMemory() - runtime.freeMemory()) + "," +
                    "native=" + formatter.format(Pointer.physicalBytes() - runtime.totalMemory())
        )
        println("$meter")
        meter.reset()
      }
      
    }
    println("Saving model to $model_file_path")
    val result = eval(act_vars)
    saveModel(model_file_path, {
      buildAct(makeObsPh = ::makeObservationPlaceholder,
               qFunc = q_func,
               numActions = env.action_space.n,
               scope = "deepq",
               reuse = ReuseOrCreateNew)
    }, act_vars.map { it.name }.zip(result))
  }
}

private fun MutableList<Float>.mean(start: Int, end: Int): Float {
  val start = kotlin.math.max(0, if (start < 0) size + start else start)
  var end = end - 1
  end = kotlin.math.max(0, if (end < 0) size + end else end)
  var sum = 0f
  for (i in start..end) {
    sum += this[i]
  }
  return sum / (end - start + 1)
}
