package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.Rand
import wumo.sim.util.d
import wumo.sim.util.ndarray.NDArray

class Box(val low: NDArray<Double>, val high: NDArray<Double>) : Space<NDArray<Double>> {
  
  init {
    require(low.size == high.size)
  }
  
  override fun sample() = NDArray(d(low.size) {
    Rand().nextDouble(low[it], high[it])
  })
  
  override fun contains(x: NDArray<Double>): Boolean {
    if (x.size != low.size) return false
    for (i in 0 until low.size)
      if (x[i] !in low[i]..high[i])
        return false
    return true
  }
}