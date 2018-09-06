package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray

class Box private constructor(
    val low: NDArray<Float>,
    val high: NDArray<Float>,
    shape: Shape,
    dataType: Class<*>)
  : Space<NDArray<Float>>(shape, dataType) {
  
  override val n = low.size
  
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
  
  companion object {
    operator fun invoke(low: NDArray<Float>, high: NDArray<Float>): Box {
      val shape = low.shape
      val dataType = low.dtype
      return Box(low, high, shape, dataType)
    }
  }
}