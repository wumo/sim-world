package wumo.sim.world.examples.algorithm

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor

private const val MAXIMUM_CAPACITY = 1 shl 30

class SuttonTileCoding(numTilesPerTiling: Int, _numTilings: Int, val allowCollisions: Boolean = false,
                       val _tiles: (DoubleArray, Int, (DoubleArray, IntArray) -> IntArray) -> IntArray) {
  val numTilings = tableSizeFor(_numTilings)
  val numOfComponents = numTilings * (numTilesPerTiling + 1)
  operator fun invoke(s: DoubleArray, a: Int): IntArray {
    return _tiles(s, a, ::tiles)
  }
  
  val data = HashMap<ArrayList<Double>, Int>(ceil(numOfComponents / 0.75).toInt())
  
  fun tiles(floats: DoubleArray, ints: IntArray): IntArray {
    for ((i, v) in floats.withIndex())
      floats[i] = floor(v * numTilings)
    
    val result = IntArray(numTilings)
    for (tiling in 0 until numTilings) {
      val tilingX2 = tiling * 2
      val coords = ArrayList<Double>(1 + floats.size + ints.size)
      coords.add(tiling.toDouble())
      var b = tiling
      for (q in floats) {
        coords.add(floor(((q + b) / numTilings)))
        b += tilingX2
      }
      for (int in ints)
        coords.add(int.toDouble())
      when {
        data.size < numOfComponents -> result[tiling] = data.getOrPut(coords, { data.size })
        allowCollisions -> result[tiling] = abs(coords.hashCode()) % numOfComponents
        else -> throw RuntimeException("too many features!")
      }
    }
    return result
  }
  
  /** Returns a power of two size for the given target capacity.*/
  fun tableSizeFor(cap: Int): Int {
    var n = cap - 1
    n = n or n.ushr(1)
    n = n or n.ushr(2)
    n = n or n.ushr(4)
    n = n or n.ushr(8)
    n = n or n.ushr(16)
    return if (n < 0) 1 else if (n >= MAXIMUM_CAPACITY) MAXIMUM_CAPACITY else n + 1
  }
}