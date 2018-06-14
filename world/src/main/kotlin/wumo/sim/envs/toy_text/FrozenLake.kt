package wumo.sim.envs.toy_text

import wumo.sim.util.ANSI_BLACK
import wumo.sim.util.ANSI_RED_BACKGROUND
import wumo.sim.util.ANSI_RESET
import wumo.sim.util.ANSI_WHITE
import wumo.sim.util.tuples.tuple2
import wumo.sim.util.tuples.tuple4

val LEFT = 0
val DOWN = 1
val RIGHT = 2
val UP = 3

val MAPS = mapOf(
    "4x4" to arrayOf(
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ),
    "8x8" to arrayOf(
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    )
)

val textForAction = arrayOf("Left", "Down", "Right", "Up")

class FrozenLake(_desc: Array<String>? = null,
                 map_name: String? = "4x4",
                 is_slippery: Boolean = true) : DiscreteEnv() {
  val desc = when {
    _desc != null -> _desc
    map_name != null -> MAPS[map_name]!!
    else -> throw Exception("Must provide either desc or map_name")
  }
  val nrow = desc.size
  val ncol = desc[0].length
  override val nS = nrow * ncol
  override val nA = 4
  override val P = Array(nS) { Array(nA) { mutableListOf<tuple4<Double, Int, Double, Boolean>>() } }
  override val isd = DoubleArray(nS) { if (desc[it / ncol][it % ncol] == 'S') 1.0 else 0.0 }
  
  init {
    fun to_s(row: Int, col: Int) = row * ncol + col
    
    fun inc(row: Int, col: Int, a: Int): tuple2<Int, Int> {
      return when (a) {
        0 -> tuple2(row, maxOf(col - 1, 0))
        1 -> tuple2(minOf(row + 1, nrow - 1), col)
        2 -> tuple2(row, minOf(col + 1, ncol - 1))
        3 -> tuple2(maxOf(row - 1, 0), col)
        else -> throw Exception("invalid action")
      }
    }
    for (row in 0 until nrow)
      for (col in 0 until ncol) {
        val s = to_s(row, col)
        for (a in 0 until 4) {
          val li = P[s][a]
          val letter = desc[row][col]
          if (letter in "GH")
            li += tuple4(1.0, s, 0.0, true)
          else {
            if (is_slippery) {
              for (b in intArrayOf(((a - 1) % 4 + 4) % 4, a, (a + 1) % 4)) {
                val (newrow, newcol) = inc(row, col, b)
                val newstate = to_s(newrow, newcol)
                val newletter = desc[newrow][newcol]
                val done = newletter in "GH"
                val rew = if (newletter == 'G') 1.0 else 0.0
                li += tuple4(1.0 / 3.0, newstate, rew, done)
              }
            } else {
              val (newrow, newcol) = inc(row, col, a)
              val newstate = to_s(newrow, newcol)
              val newletter = desc[newrow][newcol]
              val done = newletter in "GH"
              val rew = if (newletter == 'G') 1.0 else 0.0
              li += tuple4(1.0, newstate, rew, done)
            }
          }
        }
      }
  }
  
  override val reward_range = tuple2(0.0, 1.0)
  
  override fun render() {
    val row = s / ncol
    val col = s % ncol
    for ((r, line) in desc.withIndex())
      if (r == row) {
        for ((c, letter) in line.withIndex())
          if (c == col)
            print("$ANSI_RED_BACKGROUND$letter$ANSI_RESET")
          else
            print(letter)
        println()
      } else
        println(line)
    if (lastAction != -1)
      println(textForAction[lastAction])
    else
      println()
    println()
  }
  
  override fun close() {
  }
  
  override fun seed() {
  }
  
}