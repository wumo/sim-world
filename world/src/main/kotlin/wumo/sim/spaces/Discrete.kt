package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.Rand
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.types.NDInt
import wumo.sim.util.scalarDimension

class Discrete(override val n: Int) : Space<Int, Int>(NDInt, scalarDimension) {
  override fun sample() = Rand().nextInt(n)
  override fun contains(x: Int) = x in 0..(n - 1)
}