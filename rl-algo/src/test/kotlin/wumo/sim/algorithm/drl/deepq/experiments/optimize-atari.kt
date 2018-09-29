package wumo.sim.algorithm.drl.deepq.experiments

import org.junit.Test
import wumo.sim.algorithm.drl.common.make_atari
import wumo.sim.algorithm.drl.common.wrap_atari_dqn
import wumo.sim.algorithm.drl.common.wrap_deepmind
import wumo.sim.envs.envs

class optimize_atari {
  @Test
  fun test1() {
    val _env = make_atari("PongNoFrameskip-v4")
    val env = wrap_atari_dqn(_env)
    env.reset()
    var total_time = 0L
    val n = 10000
    for (i in 0 until n) {
      val start = System.nanoTime()
      val (obs, reward, done, _) = env.step(3)
      val end = System.nanoTime()
      total_time += end - start
      if (done)
        env.reset()
    }
    val s = total_time / 1e9
    println("total:$s avg:${s / n}")
  }
  
  @Test
  fun test2() {
    val env = envs.Atari("PongNoFrameskip-v4")
    env.reset()
    var total_time = 0L
    val n = 100000
    for (i in 0 until n) {
      val start = System.nanoTime()
      val (obs, reward, done, _) = env.step(3)
      val end = System.nanoTime()
      total_time += end - start
      if (done)
        env.reset()
    }
    val s = total_time / 1e9
    println("total:$s avg:${s / n}")
  }
  
  @Test
  fun test3() {
    val _env = make_atari("PongNoFrameskip-v4")
    val env = wrap_deepmind(_env, true, true, false, true)
    env.reset()
    var total_time = 0L
    val n = 10000
    for (i in 0 until n) {
      val start = System.nanoTime()
      val (obs, reward, done, _) = env.step(3)
      val end = System.nanoTime()
      total_time += end - start
      if (done)
        env.reset()
    }
    val s = total_time / 1e9
    println("total:$s avg:${s / n}")
  }
}