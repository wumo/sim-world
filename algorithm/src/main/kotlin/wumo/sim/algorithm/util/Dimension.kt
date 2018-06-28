@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.algorithm.util

infix fun Int.x(a: Int): Dimension {
  require(this >= 0 && a >= 0)
  return Dimension(mutableListOf(this.toLong(), a.toLong()))
}

infix fun Dimension.x(a: Int): Dimension {
  elements += a.toLong()
  return this
}

inline fun <T : Number> dim(d: T) = Dimension(mutableListOf(d.toLong()))

class Dimension(val elements: MutableList<Long>) : Iterable<Long> {
  fun asLongArray(): LongArray {
    return elements.toLongArray()
  }
  
  fun rank(): Long {
    return elements.size.toLong()
  }
  
  val firstDim = elements[0]
  
  val otherDim = LongArray(elements.size - 1) { elements[it + 1] }
  
  fun numElements() = elements.reduce { num, e ->
    num * e
  }
  
  override fun iterator() = elements.iterator()
}