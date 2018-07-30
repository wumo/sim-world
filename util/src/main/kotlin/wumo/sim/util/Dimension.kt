@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

infix fun Int.x(a: Int): Dimension {
  require(this >= 0 && a >= 0)
  return Dimension(mutableListOf(this, a))
}

infix fun Dimension.x(a: Int): Dimension {
  elements += a
  return this
}

infix fun Int.x(shape: Dimension) = Dimension().apply {
  elements += this@x
  elements.addAll(shape.elements)
}

inline fun <T : Number> dim(d: T) = Dimension(mutableListOf(d.toInt()))

val scalarDimension = Dimension()

class Dimension(val elements: MutableList<Int> = mutableListOf()) : Iterable<Int> {
  constructor(elements: LongArray) : this(MutableList(elements.size) { elements[it].toInt() })
  constructor(elements: IntArray) : this(elements.toMutableList())
  
  fun asLongArray() =
      LongArray(elements.size) { elements[it].toLong() }
  
  fun asIntArray() =
      IntArray(elements.size) { elements[it] }
  
  fun rank(): Int {
    return elements.size
  }
  
  val is_fully_defined: Boolean
    get() {
      for (element in elements)
        if (element < 0) return false
      return true
    }
  
  fun numElements() = if (elements.isEmpty()) 1 else elements.reduce { num, e ->
    num * e
  }
  
  operator fun get(idx: Int): Int {
    return if (idx < 0)
      elements[elements.size + idx]
    else
      elements[idx]
  }
  
  override fun iterator() = elements.iterator()
  
  override fun toString(): String {
    val sb = StringBuilder()
    sb.append("(")
    for ((i, value) in elements.withIndex()) {
      sb.append(if (value == -1) "?" else value)
      if (i < elements.lastIndex)
        sb.append(", ")
    }
    sb.append(")")
    return sb.toString()
  }
  
  fun isCompatibleWith(other: Dimension): Boolean {
    if (rank() != other.rank()) return false
    for (i in 0 until rank()) {
      if (!compatible(this[i], other[i]))
        return false
    }
    return true
  }
  
  fun compatible(d1: Int, d2: Int) = when {
    d1 == d2 -> true
    d1 == -1 || d2 == -1 -> true
    else -> false
  }
}