@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

inline fun String.print(suffix: String) = kotlin.io.print(this + suffix)
inline fun String.println(suffix: String) = kotlin.io.println(this + suffix)
inline fun String.println() = kotlin.io.println(this)

inline operator fun StringBuilder.plusAssign(value: Any) {
  append(value)
}

class StringBuilderEx {
  val sb = StringBuilder()
  inline operator fun Any?.unaryPlus(): StringBuilderEx {
    if (this != null) append(this)
    return this@StringBuilderEx
  }
  
  inline operator fun plus(any: Any?): StringBuilderEx {
    if (any != null) append(any)
    return this
  }
  
  fun append(any: Any) = sb.append(any)
  
  override fun toString() = sb.toString()
}

inline fun sb(block: StringBuilderEx.(StringBuilderEx) -> Unit): String {
  val sb = StringBuilderEx()
  block(sb, sb)
  return sb.toString()
}

