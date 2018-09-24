package wumo.sim.core

import wumo.sim.util.t4
import kotlin.random.Random

open class Wrapper<O, OE : Any, A, AE : Any, WrappedENV>
constructor(val env: Env<O, OE, A, AE, WrappedENV>) : Env<O, OE, A, AE, WrappedENV> {
  
  override var rand: Random = env.rand
  override val action_space: Space<A, AE> = env.action_space
  override val observation_space: Space<O, OE> = env.observation_space
  
  override fun step(a: A): t4<O, Float, Boolean, Map<String, Any>> = env.step(a)
  
  override fun reset(): O = env.reset()
  
  override fun render() = env.render()
  
  override fun close() = env.close()
  
  override fun seed(seed: Long?): List<Long> = env.seed(seed)
  override fun toString(): String {
    return "<${this::class.simpleName}$env>"
  }
  
  override val unwrapped: WrappedENV = env.unwrapped
}