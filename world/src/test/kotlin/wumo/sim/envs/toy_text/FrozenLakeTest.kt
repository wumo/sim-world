package wumo.sim.envs.toy_text

import org.junit.Test

class FrozenLakeTest {
  @Test
  fun `test render slippery`() {
    val env = FrozenLake()
    env.reset()
    for (i in 0 until 1000) {
      env.render()
      val (s, r, d, _) = env.step(env.action_space.sample())
      if (d)
        env.reset()
    }
    env.close()
  }
  
  @Test
  fun `test render no slippery`() {
    val env = FrozenLake(is_slippery = false)
    env.reset()
    for (i in 0 until 1000) {
      env.render()
      val (s, r, d, _) = env.step(env.action_space.sample())
      if (d)
        env.reset()
    }
    env.close()
  }
}