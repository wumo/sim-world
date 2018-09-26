package wumo.sim.util

data class TimeSummary(val avg: Double, val n: Int) {
  
  override fun toString(): String {
    return "${avg * n}"
  }
}

class TimeMeter {
  val history = mutableMapOf<String, MutableList<Long>>()
  val targets = mutableMapOf<String, Long>()
  fun start(label: String) {
    require(label !in targets)
    targets[label] = System.nanoTime()
  }
  
  fun end(label: String) {
    val start = targets.remove(label)!!
    val h = history.getOrPut(label) { mutableListOf() }
    h += System.nanoTime() - start
  }
  
  fun summary(label: String): TimeSummary {
    val h = history[label]!!
    val n = h.size.toDouble() * 1e9
    val avg = h.sumByDouble { it / n }
    return TimeSummary(avg, h.size)
  }
  
  fun reset() {
    targets.clear()
    history.clear()
  }
  
  override fun toString(): String =
      history.keys.joinToString(",") {
        it + ":" + summary(it)
      }
  
}