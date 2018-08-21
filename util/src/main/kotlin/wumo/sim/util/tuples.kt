package wumo.sim.util

val <A, B> Pair<A, B>._1
  get() = first
val <A, B> Pair<A, B>._2
  get() = second

data class t2<A, B>(var _1: A, var _2: B) {
  override fun toString(): String = "($_1,$_2)"
}

data class t3<A, B, C>(var _1: A, var _2: B, var _3: C) {
  override fun toString(): String {
    return "($_1,$_2,$_3)"
  }
}

data class t4<A, B, C, D>(var _1: A, var _2: B, var _3: C, var _4: D) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4)"
  }
}

data class t5<A, B, C, D, E>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5)"
  }
}

data class t6<A, B, C, D, E, F>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E, var _6: F) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5,$_6)"
  }
}

data class t7<A, B, C, D, E, F, G>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E, var _6: F, var _7: G) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5,$_6,$_7)"
  }
}

fun <T> Iterable<T>.collectionSizeOrDefault(default: Int): Int = if (this is Collection<*>) this.size else default

//inline fun <T, R, V> Iterable<T>.zip(other: Iterable<R>, transform: (a: T, b: R) -> V): List<V> {
//  val first = iterator()
//  val second = other.iterator()
//  val list = ArrayList<V>(minOf(collectionSizeOrDefault(10), other.collectionSizeOrDefault(10)))
//  while (first.hasNext() && second.hasNext()) {
//    list.add(transform(first.next(), second.next()))
//  }
//  return list
//}

//infix fun <T, R> Iterable<T>.zip(other: Iterable<R>): List<t2<T, R>> {
//  return zip(other) { t1, t2 -> t2(t1, t2) }
//}