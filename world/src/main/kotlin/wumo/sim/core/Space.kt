package wumo.sim.core

import wumo.sim.util.Shape

abstract class Space<E>(val shape: Shape = Shape(), val dataType: Class<*> = Float::class.java) {
  abstract val n: Int
  abstract fun sample(): E
  abstract fun contains(x: E): Boolean
}