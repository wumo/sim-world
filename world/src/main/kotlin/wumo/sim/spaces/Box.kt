package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.types.NDFloat

class Box<T> constructor(
    val low: NDArray<T>,
    val high: NDArray<T>)
  : Space<NDArray<T>, T>(low.dtype, low.shape)
    where T : Number, T : Comparable<T> {
  
  override val n = low.size
  
  init {
    require(low.shape == high.shape)
  }
  
  override fun sample(): NDArray<T> = NDArray(
      shape,
      dtype.makeBuf(shape.numElements()) {
        val e = Rand().nextFloat(low[it].toFloat(),
                                 high[it].toFloat() +
                                     if (dtype == NDFloat) 0f else 1f)
        dtype.cast(e)
      }, dtype)
  
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