package wumo.sim.envs

import wumo.sim.core.Env
import wumo.sim.envs.atari.AtariEnv
import wumo.sim.envs.atari.AtariEnv.Companion.ObsType
import wumo.sim.envs.atari.AtariEnvType
import wumo.sim.envs.classic_control.CartPole
import wumo.sim.envs.classic_control.MountainCar
import wumo.sim.envs.toy_text.FrozenLake
import wumo.sim.util.errorIf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.wrappers.TimeLimit

object envs {
  fun `CartPole-v0`(max_episode_steps: Int = 200,
                    reward_threshold: Float = 195f): Env<NDArray<Float>, Int, CartPole> {
    return TimeLimit(CartPole(), max_episode_steps)
  }
  
  fun `MountainCar-v0`(max_episode_steps: Int = 200,
                       reward_threshold: Float = -110f): Env<NDArray<Float>, Int, MountainCar> {
    return TimeLimit(MountainCar(), max_episode_steps)
  }
  
  fun `FrozenLake-v0`(max_episode_steps: Int = 200,
                      reward_threshold: Float = 0.78f): Env<Int, Int, FrozenLake> {
    return TimeLimit(FrozenLake(map_name = "4x4"), max_episode_steps)
  }
  
  fun `FrozenLake8x8-v0`(max_episode_steps: Int = 200,
                         reward_threshold: Float = 0.99f): Env<Int, Int, FrozenLake> {
    return TimeLimit(FrozenLake(map_name = "8x8"), max_episode_steps)
  }
  
  private val namePattern =
      Regex("""^(?:((?:[A-Z][a-z]*)+)(-ram)?(Deterministic|NoFrameskip)-(v0|v4))|(?:((?:[A-Z][a-z]*)+)(-ram)?-(v0|v4))$""")
  private val wordPattern = Regex("[A-Z][a-z]*")
  fun Atari(id: String): AtariEnvType {
    val result = namePattern.matchEntire(id)
    errorIf(result == null) { "invalid game id:$id" }
    val (_game, _ram, mode, _version, _game1, _ram1, _version1) = result!!.destructured
    val game = (_game + _game1).replace(wordPattern) {
      it.value.toLowerCase() + "_"
    }.dropLast(1)
    val ram = _ram + _ram1
    val version = _version + _version1
    val obs_type = if (ram.isNotEmpty()) ObsType.ram
    else ObsType.image
    
    val nondeterministic = game == "elevator_action" && obs_type == ObsType.ram
    val repeat_action_probability = if (version == "v0") 0.25f else 0f
    val _frameskip = if (game == "space_invaders") 3 else 4
    val frameskip: Pair<Int, Int>
    val max_episode_steps: Int
    when (mode) {
      "Deterministic" -> {
        frameskip = _frameskip to _frameskip
        max_episode_steps = 10_0000
      }
      "NoFrameskip" -> {
        frameskip = 1 to 1
        max_episode_steps = _frameskip * 10_0000
      }
      else -> {
        frameskip = 2 to 5
        max_episode_steps = if (version == "v0") 10000 else 10_0000
      }
    }
    return TimeLimit(
        AtariEnv(game = game,
                 obs_type = obs_type,
                 frameskip = frameskip,
                 repeat_action_probability = repeat_action_probability),
        max_episode_steps)
  }
}
