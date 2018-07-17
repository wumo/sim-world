package wumo.sim.util

data class tuple2<A, B>(var _1: A, var _2: B) {
  override fun toString(): String = "($_1,$_2)"
}

data class tuple3<A, B, C>(var _1: A, var _2: B, var _3: C) {
  override fun toString(): String {
    return "($_1,$_2,$_3)"
  }
}

data class tuple4<A, B, C, D>(var _1: A, var _2: B, var _3: C, var _4: D) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4)"
  }
}

data class tuple5<A, B, C, D, E>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5)"
  }
}

data class tuple6<A, B, C, D, E, F>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E, var _6: F) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5,$_6)"
  }
}

data class tuple7<A, B, C, D, E, F, G>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E, var _6: F, var _7: G) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5,$_6,$_7)"
  }
}

fun <T> Iterable<T>.collectionSizeOrDefault(default: Int): Int = if (this is Collection<*>) this.size else default

inline fun <T, R, V> Iterable<T>.zip(other: Iterable<R>, transform: (a: T, b: R) -> V): List<V> {
  val first = iterator()
  val second = other.iterator()
  val list = ArrayList<V>(minOf(collectionSizeOrDefault(10), other.collectionSizeOrDefault(10)))
  while (first.hasNext() && second.hasNext()) {
    list.add(transform(first.next(), second.next()))
  }
  return list
}

infix fun <T, R> Iterable<T>.zip(other: Iterable<R>): List<tuple2<T, R>> {
  return zip(other) { t1, t2 -> tuple2(t1, t2) }
}