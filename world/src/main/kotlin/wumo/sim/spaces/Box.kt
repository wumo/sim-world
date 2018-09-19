package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.cast
import wumo.sim.util.ndarray.toNDArray

class Box<T> constructor(
    val low: NDArray<T>,
    val high: NDArray<T>)
  : Space<NDArray<T>>(low.shape, low.dtype)
    where T : Number, T : Comparable<T> {
  
  override val n = low.size
  
  init {
    require(low.size == high.size)
  }
  
  override fun sample(): NDArray<T> = Array(low.size) {
    Rand().nextFloat(low[it].toFloat(),
                     high[it].toFloat() +
                         if (dataType == Float::class.java) 0f else 1f)
        .cast(dataType as Class<T>) as Any
  }.toNDArray()
  
  override fun contains(x: NDArray<T>): Boolean {
    if (x.size != low.size) return false
    for (i in 0 until low.size)
      if (x[i] !in low[i]..high[i])
        return false
    return true
  }
  
  companion object {
    inline operator fun <reified T>
        invoke(low: T, high: T, shape: Shape): Box<T>
        where T : Number, T : Comparable<T> {
      val low = NDArray(shape, low)
      val high = NDArray(shape, high)
      return Box(low, high)
    }
  }
}