package wumo.sim.envs.atari

import org.junit.Assert.*
import org.junit.Test
import wumo.sim.envs.atari.AtariEnv.Companion.ObsType.image

class AtariEnvTest {
  @Test
  fun test() {
    
    val env = AtariEnv(game = "space_invaders")
    
    val episode = 10
    repeat(episode) {
      env.reset()
      var done = false
      var reward = 0.0
      while (!done) {
        env.render()
        val a = env.action_space.sample()
        val (ob, _reward, _done, _) = env.step(a)
        reward += _reward
        done = _done
      }
      println(reward)
    }
    env.close()
  }
}