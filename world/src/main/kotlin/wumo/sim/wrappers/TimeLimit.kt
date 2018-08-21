package wumo.sim.wrappers

import wumo.sim.core.Env
import wumo.sim.util.t4

class TimeLimit<O : Any, A : Any>(val env: Env<O, A>,
                                  val max_episode_steps: Int? = null,
                                  val max_episode_seconds: Int? = null) : Env<O, A> {
  var elapsed_steps = 0
  var episode_started_at: Long = 0
  
  val elapsed_seconds
    get() = (System.currentTimeMillis() - episode_started_at) / 1000
  
  fun past_limit() = when {
    (max_episode_steps != null && max_episode_steps <= elapsed_steps) ||
    (max_episode_seconds != null && max_episode_seconds <= elapsed_seconds) ->
      true
    else ->
      false
  }
  
  override val action_space = env.action_space
  override val observation_space = env.observation_space
  
  override fun step(a: A): t4<O, Float, Boolean, Map<String, Any>> {
    val tuple = env.step(a)
    elapsed_steps++
    if (past_limit())
      tuple._3 = true
    return tuple
  }
  
  override fun reset(): O {
    episode_started_at = System.currentTimeMillis()
    elapsed_steps = 0
    return env.reset() as O
  }
  
  override fun render() {
    env.render()
  }
  
  override fun close() {
    env.close()
  }
  
  override fun seed() {
    env.seed()
  }
}