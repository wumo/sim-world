package wumo.sim.algorithm.drl.common

import org.junit.Test

class TestAtariWrappers {
  @Test
  fun testWrapDeepmind() {
    val _env = make_atari("PongNoFrameskip-v4")
    val env = wrap_atari_dqn(_env)
    
    val episode = 10
    var i = 0
    repeat(episode) {
      env.reset()
      var done = false
      var reward = 0.0
      
      while (!done) {
        env.render()
        val a = env.action_space.sample()
        val (ob, _reward, _done, _) = env.step(a)
//        println(i++)
        reward += _reward
        done = _done
      }
      println(reward)
    }
    env.close()
  }
}