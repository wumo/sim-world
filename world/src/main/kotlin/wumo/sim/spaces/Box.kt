package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray

class Box(val low: NDArray<Float>, val high: NDArray<Float>) : Space<NDArray<Float>> {
  override val n = low.size
  val shape: Dimension = low.shape
  val dtype: Int = 1
  
  init {
    require(low.size == high.size)
  }
  
  override fun sample() = NDArray(f(low.size) {
    Rand().nextFloat(low[it], high[it])
  })
  
  override fun contains(x: NDArray<Float>): Boolean {
    if (x.size != low.size) return false
    for (i in 0 until low.size)
      if (x[i] !in low[i]..high[i])
        return false
    return true
  }
}