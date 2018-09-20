package wumo.sim.util

import java.lang.Exception
import kotlin.contracts.*

/**
 * Always throws [NotImplementedError] stating that operation is not implemented.
 */

@Suppress("NOTHING_TO_INLINE")
inline fun NONE(): Nothing = throw NotImplementedError()

inline fun errorIf(prop: Boolean, message: () -> String) {
  if (prop)
    throw Exception(message())
}