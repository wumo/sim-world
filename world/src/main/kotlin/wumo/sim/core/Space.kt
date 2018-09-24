package wumo.sim.core

import wumo.sim.util.Shape
import wumo.sim.util.ndarray.types.NDType

abstract class Space<E, T : Any>(val dtype: NDType<T>,
                                 val shape: Shape = Shape()) {
  abstract val n: Int
  abstract fun sample(): E
  abstract fun contains(x: E): Boolean
}