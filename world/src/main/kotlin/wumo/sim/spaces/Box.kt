package wumo.sim.spaces

import wumo.sim.core.Space
import wumo.sim.util.Rand
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.types.NDFloat
import wumo.sim.util.ndarray.types.NDType
import wumo.sim.util.nextFloat

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
      })
  
  override fun contains(x: NDArray<T>): Boolean {
    if (x.size != low.size) return false
    for (i in 0 until low.size)
      if (x[i] !in low[i]..high[i])
        return false
    return true
  }
  
  companion object {
    operator fun <T> invoke(low: T, high: T, shape: Shape, dtype: NDType<T>): Box<T>
        where T : Number, T : Comparable<T> {
      val low = NDArray(shape, dtype) { low }
      val high = NDArray(shape, dtype) { high }
      return Box(low, high)
    }
  }
}