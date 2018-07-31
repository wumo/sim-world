package wumo.sim.envs

import wumo.sim.envs.classic_control.CartPole
import wumo.sim.envs.classic_control.MountainCar

fun register_envs() {
  register("CartPole-v0", { CartPole() },
           max_episode_steps = 200,
           reward_threshold = 195f)
  register("MountainCar-v0", { MountainCar() },
           max_episode_steps = 200,
           reward_threshold = -110f)
}