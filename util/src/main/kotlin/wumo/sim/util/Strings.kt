@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

inline fun String.print(suffix: String) = kotlin.io.print(this + suffix)
inline fun String.println(suffix: String) = kotlin.io.println(this + suffix)
inline fun String.println() = kotlin.io.println(this)

inline operator fun StringBuilder.plusAssign(value: Any) {
  append(value)
}
