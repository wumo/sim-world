package wumo.sim.algorithm.drl.common

open class SegmentTree<T>(val capacity: Int,
                          inline val operation: (T, T) -> T,
                          neutral_element: T) {
  
  init {
    assert(capacity > 0 && (capacity and (capacity - 1)) == 0) {
      "capacity must be positive and a power of 2."
    }
  }
  
  val value = MutableList(2 * capacity) { neutral_element }
  
  fun reduce_helper(start: Int, end: Int, node: Int, node_start: Int, node_end: Int): T {
    if (start == node_start && end == node_end)
      return value[node]
    val mid = (node_start + node_end) / 2
    return when {
      end <= mid -> reduce_helper(start, end, 2 * node, node_start, mid)
      mid + 1 <= start -> reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
      else -> operation(
          reduce_helper(start, mid, 2 * node, node_start, mid),
          reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
      )
    }
  }
  
  fun reduce(start: Int = 0, end: Int? = null): T {
    var end = end ?: capacity
    if (end < 0)
      end += capacity
    end -= 1
    return reduce_helper(start, end, 1, 0, capacity - 1)
  }
  
  operator fun set(idx: Int, v: T) {
    //index of the leaf
    var idx = idx + capacity
    value[idx] = v
    idx /= 2
    while (idx >= 1) {
      value[idx] = operation(
          value[2 * idx],
          value[2 * idx + 1]
      )
      idx /= 2
    }
  }
  
  operator fun get(idx: Int): T {
    assert(idx in 0 until capacity)
    return value[capacity + idx]
  }
}

class SumSegmentTree(capacity: Int) : SegmentTree<Float>(
    capacity = capacity,
    operation = { a, b -> a + b },
    neutral_element = 0f) {
  
  fun sum(start: Int = 0, end: Int? = null): Float =
      reduce(start, end)
  
  fun find_prefixsum_idx(prefixsum: Float): Int {
    assert(prefixsum in 0f..(sum() + 1e-5f))
    var prefixsum = prefixsum
    var idx = 1
    while (idx < capacity)//while non-leaf
      if (value[2 * idx] > prefixsum)
        idx *= 2
      else {
        prefixsum -= value[2 * idx]
        idx = 2 * idx + 1
      }
    return idx - capacity
  }
}

class MinSegmentTree(capacity: Int) : SegmentTree<Float>(
    capacity = capacity,
    operation = { a, b -> minOf(a, b) },
    neutral_element = Float.POSITIVE_INFINITY) {
  
  fun min(start: Int = 0, end: Int? = null): Float =
      reduce(start, end)
}