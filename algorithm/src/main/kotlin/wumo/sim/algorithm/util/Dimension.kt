@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.algorithm.util

infix fun Int.x(a: Int): Dimension {
  require(this >= 0 && a >= 0)
  return Dimension(mutableListOf(this, a))
}

infix fun Dimension.x(a: Int): Dimension {
  elements += a
  return this
}

inline fun <T : Number> dim(d: T) = Dimension(mutableListOf(d.toInt()))

internal val scalarDimension = Dimension()

class Dimension(val elements: MutableList<Int> = mutableListOf()) : Iterable<Int> {
  constructor(elements: LongArray) : this(MutableList(elements.size) { elements[it].toInt() })
  
  fun asLongArray(): LongArray {
    return LongArray(elements.size) { elements[it].toLong() }
  }
  
  fun rank(): Int {
    return elements.size
  }
  
  val firstDim
    get() = elements[0]
  
  val otherDim
    get() = LongArray(elements.size - 1) { elements[it + 1].toLong() }
  
  fun numElements() = elements.reduce { num, e ->
    num * e
  }
  
  operator fun get(idx: Int): Int {
    return if (idx < 0)
      elements[elements.size + idx]
    else
      elements[idx]
  }
  
  override fun iterator() = elements.iterator()
  
  override fun toString() = elements.toString()
}