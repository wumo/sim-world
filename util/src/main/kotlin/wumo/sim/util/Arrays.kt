@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

inline fun f(vararg elements: Float) = elements
inline fun d(vararg elements: Double) = elements
inline fun B(vararg elements: Boolean) = elements
inline fun b(vararg elements: Byte) = elements
inline fun s(vararg elements: Short) = elements
inline fun i(vararg elements: Int) = elements
inline fun l(vararg elements: Long) = elements
inline fun <reified T> a(vararg elements: T) = elements as Array<T>

fun arange(stop: Int) = IntArray(stop) { it }

inline fun arrayCopy(src: Any, dst: Any, n: Int) {
  System.arraycopy(src, 0, dst, 0, n)
}