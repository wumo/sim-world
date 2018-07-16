package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.Rand

class Box(val low: DoubleArray, val high: DoubleArray) : Space<DoubleArray> {
  
  init {
    require(low.size == high.size)
  }
  
  override fun sample() = DoubleArray(low.size) {
    Rand().nextDouble(low[it], high[it])
  }
  
  override fun contains(x: DoubleArray): Boolean {
    if (x.size != low.size) return false
    for (i in 0 until low.size)
      if (x[i] !in low[i]..high[i])
        return false
    return true
  }
}