@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util.collections

inline fun arrayCopy(src: Any, dst: Any, n: Int) {
  System.arraycopy(src, 0, dst, 0, n)
}