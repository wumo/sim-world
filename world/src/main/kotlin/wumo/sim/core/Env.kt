package wumo.sim.core

import wumo.sim.util.t2
import wumo.sim.util.t4
import kotlin.random.Random

interface Env<O, A, WrappedENV> {
  val reward_range: t2<Float, Float>
    get() = t2(Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY)
  var rand: Random
  val action_space: Space<A>
  val observation_space: Space<O>
  fun step(a: A): t4<O, Float, Boolean, Map<String, Any>>
  fun reset(): O
  fun render()
  fun close()
  fun seed(seed: Long? = null): List<Long>
  
  val unwrapped: WrappedENV
    get() = this as WrappedENV
}