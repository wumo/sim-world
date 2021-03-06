@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

import wumo.sim.util.ndarray.NDArray
import java.util.concurrent.ThreadLocalRandom
import kotlin.random.Random

inline fun Rand() = ThreadLocalRandom.current()!!
inline fun ThreadLocalRandom.nextFloat(origin: Float, bound: Float) = origin + nextFloat() * (bound - origin)
fun ThreadLocalRandom.nextGaussian(mean: Float = 0f, deviation: Float = 1f): Float =
    nextGaussian().toFloat() * deviation + mean

fun Rand(low: Float, high: Float, n: Int) = NDArray(f(n) { Rand().nextFloat(low, high) })

fun Random.uniform(low: Float, high: Float, n: Int): NDArray<Float> =
    NDArray(f(n) {
      this@uniform.nextDouble(low.toDouble(),
                              high.toDouble()).toFloat()
    })

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

fun <T, N> argmax(set: Iterable<T>, evaluate: (T) -> N): T
    where N : Number, N : Comparable<N> {
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

fun <T, N> argmin(set: Iterable<T>, evaluate: (T) -> N): T
    where N : Number, N : Comparable<N> {
  val iterator = set.iterator()
  val min_a = mutableListOf(iterator.next())
  var min = evaluate(min_a[0])
  while (iterator.hasNext()) {
    val tmp = iterator.next()
    val p = evaluate(tmp)
    if (p < min) {
      min = p
      min_a.apply {
        clear()
        add(tmp)
      }
    } else if (p == min)
      min_a.add(tmp)
  }
  return min_a[Rand().nextInt(min_a.size)]
}