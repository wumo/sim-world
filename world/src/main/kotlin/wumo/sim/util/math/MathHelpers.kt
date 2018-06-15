@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util.math

import java.util.concurrent.ThreadLocalRandom

inline fun Rand() = ThreadLocalRandom.current()!!
fun Rand(low: Double, high: Double, n: Int) = DoubleArray(n) { Rand().nextDouble(low, high) }
operator fun DoubleArray.unaryMinus() =
    DoubleArray(this.size) {
      -this[it]
    }

fun <T> max(set: Iterable<T>, default: Double = Double.NaN, evaluate: T.(T) -> Double): Double {
  val iterator = set.iterator()
  if (!iterator.hasNext()) return default
  var tmp = iterator.next()
  var max = evaluate(tmp, tmp)
  while (iterator.hasNext()) {
    tmp = iterator.next()
    val p = evaluate(tmp, tmp)
    if (p > max)
      max = p
  }
  return max
}

fun <T> argmax(set: Iterable<T>, evaluate: (T) -> Double): T {
  val iterator = set.iterator()
  val max_a = mutableListOf(iterator.next())
  var max = evaluate(max_a[0])
  while (iterator.hasNext()) {
    val tmp = iterator.next()
    val p = evaluate(tmp)
    if (p > max) {
      max = p
      max_a.apply {
        clear()
        add(tmp)
      }
    } else if (p == max)
      max_a.add(tmp)
  }
  return max_a[Rand().nextInt(max_a.size)]
}