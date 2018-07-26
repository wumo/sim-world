package wumo.sim.util

class ClosedFloatRange(val start: Float, val last: Float)
  : Iterable<Float> {
  var step = last - start
  
  init {
    check(start < last && step > 0)
  }
  
  override fun iterator() = object : Iterator<Float> {
    var current = start
    override fun hasNext() = current < last
    
    override fun next(): Float {
      current += step
      return minOf(last, current)
    }
  }
  
  operator fun div(n: Int): ClosedFloatRange {
    check(n > 0)
    this.step = (last - start) / n
    return this
  }
}

operator fun Float.rangeTo(endInclusive: Float) =
    ClosedFloatRange(this, endInclusive)