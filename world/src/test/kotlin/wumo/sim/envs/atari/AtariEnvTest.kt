package wumo.sim.envs.atari

import org.bytedeco.javacpp.Pointer
import org.junit.Test
import wumo.sim.envs.envs
import wumo.sim.util.native
import java.text.NumberFormat

class AtariEnvTest {
  
  @Test
  fun make() {
    val a = envs.Atari("SpaceInvaders-ram-v4")
  }
  
  val formatter = NumberFormat.getInstance()
  @Test
  fun test() {
    
    val env = envs.Atari("PongNoFrameskip-v4")
    
    val episode = 10000
    var i = 0
    repeat(episode) {
      native {
        env.reset()
        var done = false
        var reward = 0.0
        
        while (!done) {
//        env.render()
          val a = env.action_space.sample()
          native {
            val (ob, _reward, _done, _) = env.step(a)
            reward += _reward
            done = _done
          }
        }
        println(formatter.format(Pointer.physicalBytes()))
      }
    }
    env.close()
  }
}