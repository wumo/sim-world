package wumo.sim.util

/**
 * Groups elements of the original collection by the key returned by the given [keySelector] function
 * applied to each element and returns a map where each group key is associated with a list of corresponding elements.
 *
 * The returned map preserves the entry iteration order of the keys produced from the original collection.
 *
 * @sample samples.collections.Collections.Transformations.groupBy
 */
inline fun <T, K> Iterable<T>.groupBy(keySelector: (T) -> K): Map<K, MutableSet<T>> {
  return groupByTo(LinkedHashMap(), keySelector)
}

/**
 * Groups elements of the original collection by the key returned by the given [keySelector] function
 * applied to each element and puts to the [destination] map each group key associated with a list of corresponding elements.
 *
 * @return The [destination] map.
 *
 * @sample samples.collections.Collections.Transformations.groupBy
 */
inline fun <T, K, M : MutableMap<in K, MutableSet<T>>> Iterable<T>.groupByTo(destination: M, keySelector: (T) -> K): M {
  for (element in this) {
    val key = keySelector(element)
    val list = destination.getOrPut(key) { mutableSetOf() }
    list.add(element)
  }
  return destination
}