package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.Rand

class Discrete(val n: Int) : Space<Int> {
  override fun sample() = Rand().nextInt(n)
  override fun contains(x: Int) = x in 0..(n - 1)
}