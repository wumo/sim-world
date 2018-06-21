package wumo.sim.algorithm.util

infix fun Int.x(a: Int): Dimension {
  require(this >= 0 && a >= 0)
  return Dimension(mutableListOf(this.toLong(), a.toLong()))
}

infix fun Dimension.x(a: Int): Dimension {
  elements += a.toLong()
  return this
}

class Dimension(val elements: MutableList<Long>) {
  fun asLongArray(): LongArray {
    return elements.toLongArray()
  }
  
  fun rank(): Long {
    return elements.size.toLong()
  }
}