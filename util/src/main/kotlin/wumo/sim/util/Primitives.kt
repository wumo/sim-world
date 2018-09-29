@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

const val byte0: Byte = 0

inline fun Boolean.toByte(): Byte = if (this) 1 else 0

inline fun Byte.toBool(): Boolean = this != byte0