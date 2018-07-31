package wumo.sim.envs

import wumo.sim.core.Env
import wumo.sim.wrappers.TimeLimit

/**
A specification for a particular instance of the environment. Used
to register the parameters for official evaluations.

Args:
id (str): The official environment ID
entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
trials (int): The number of trials to average reward over
reward_threshold (Optional[int]): The reward threshold before the task is considered solved
local_only: True iff the environment is to be used only on the local machine (e.g. debugging envs)
kwargs (dict): The kwargs to pass to the environment class
nondeterministic (bool): Whether this environment is non-deterministic even after seeding
tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags

Attributes:
id (str): The official environment ID
trials (int): The number of trials run in official evaluation
 */
class EnvSpec(val id: String,
              val maker: () -> Env<*, *>,
              val trials: Int = 100,
              val nondeterministic: Boolean = false,
              val reward_threshold: Float? = null,
              val max_episode_steps: Int? = null,
              val max_episode_seconds: Int? = null) {
  val timestep_limit
    get() = max_episode_steps
  
  fun make(): Env<*, *> {
    return maker()
  }
}

class EnvRegistry {
  val env_specs = hashMapOf<String, EnvSpec>()
  fun make(id: String): Env<*, *> {
    val spec = env_specs[id]!!
    var env = spec.make()
    val timestep_limit = spec.timestep_limit
    if (timestep_limit != null)
      env = TimeLimit(env, max_episode_steps = spec.max_episode_steps,
                      max_episode_seconds = spec.max_episode_seconds)
    return env
  }
  
  fun register(id: String, maker: () -> Env<*, *>, max_episode_steps: Int, reward_threshold: Float) {
    env_specs[id] = EnvSpec(id, maker, max_episode_steps = max_episode_steps, reward_threshold = reward_threshold)
  }
}

val gym = EnvRegistry()
fun register(id: String, maker: () -> Env<*, *>, max_episode_steps: Int, reward_threshold: Float) {
  gym.register(id, maker, max_episode_steps, reward_threshold)
}