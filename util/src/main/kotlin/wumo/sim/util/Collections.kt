@file:Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")

package wumo.sim.util

import java.util.*

inline fun f(vararg elements: Float) = elements
inline fun f(size: Int, init: (Int) -> Float) = FloatArray(size) { init(it) }
inline fun d(vararg elements: Double) = elements
inline fun d(size: Int, init: (Int) -> Double) = DoubleArray(size) { init(it) }
inline fun B(vararg elements: Boolean) = elements
inline fun B(size: Int, init: (Int) -> Boolean) = BooleanArray(size) { init(it) }
inline fun b(vararg elements: Byte) = elements
inline fun b(size: Int, init: (Int) -> Byte) = ByteArray(size) { init(it) }
inline fun s(vararg elements: Short) = elements
inline fun s(size: Int, init: (Int) -> Short) = ShortArray(size) { init(it) }
inline fun i(vararg elements: Int) = elements
inline fun i(size: Int, init: (Int) -> Int) = IntArray(size) { init(it) }
inline fun l(vararg elements: Long) = elements
inline fun l(size: Int, init: (Int) -> Long) = LongArray(size) { init(it) }
inline fun <reified T> a(vararg elements: T) = elements as Array<T>
inline fun <reified T> a(size: Int, init: (Int) -> T) = Array(size) { init(it) }
fun arange(stop: Int) = IntArray(stop) { it }

inline fun arrayCopy(src: Any, dst: Any, n: Int) {
  System.arraycopy(src, 0, dst, 0, n)
}

inline fun <R> emptyMutableSet(): MutableSet<R> = Collections.emptySet<R>()

inline fun <K, V> emptyMutableMap(): MutableMap<K, V> = Collections.emptyMap()

inline fun <E> MutableCollection<E>.append(vararg elements: E) {
  for (element in elements)
    add(element)
}

inline fun <E> Iterable<E>.firstAndRest(blockFirst: (E) -> Unit, blockRest: (E) -> Unit) {
  val iterator = iterator()
  val i = iterator.next()
  blockFirst(i)
  while (iterator.hasNext())
    blockRest(iterator.next())
}

inline fun <R1, R2, R3> zip(s1: Iterable<R1>,
                            s2: Iterable<R2>,
                            s3: Iterable<R3>,
                            block: (t3<R1, R2, R3>) -> Unit): Unit {
  val i1 = s1.iterator()
  val i2 = s2.iterator()
  val i3 = s3.iterator()
  while (i1.hasNext() && i2.hasNext() && i3.hasNext()) {
    block(t3(i1.next(), i2.next(), i3.next()))
  }
}