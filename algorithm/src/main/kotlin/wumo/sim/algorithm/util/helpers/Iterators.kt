package wumo.sim.algorithm.util.helpers

inline fun <T, E> loop(crossinline init: () -> T,
                       crossinline continueCond: (T) -> Boolean,
                       crossinline increment: (T) -> Unit,
                       crossinline element: (T) -> E) = object : Iterator<E> {
  val iter = init()
  
  override fun hasNext() = continueCond(iter)
  
  override fun next(): E {
    val e = element(iter)
    increment(iter)
    return e
  }
}