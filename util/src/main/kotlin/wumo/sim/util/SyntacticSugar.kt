package wumo.sim.util

inline fun <R> expr(block: () -> R) = block()

/**
 * Always throws [NotImplementedError] stating that operation is not implemented.
 */

@Suppress("NOTHING_TO_INLINE")
inline fun NOPE(): Nothing = throw NotImplementedError()