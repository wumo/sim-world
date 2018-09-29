package wumo.sim.envs.toy_text

import wumo.sim.util.*

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

class FrozenLake
private constructor(val desc: Array<String>, val nrow: Int, val ncol: Int,
                    nS: Int, nA: Int,
                    P: Transition,
                    isd: FloatArray) : DiscreteEnv<FrozenLake>(nS, nA, P, isd) {
  
  companion object {
    operator fun invoke(_desc: Array<String>? = null,
                        map_name: String? = "4x4",
                        is_slippery: Boolean = true): FrozenLake {
      val desc = when {
        _desc != null -> _desc
        map_name != null -> MAPS[map_name]!!
        else -> throw Exception("Must provide either desc or map_name")
      }
      val nrow = desc.size
      val ncol = desc[0].length
      val nS = nrow * ncol
      val nA = 4
      val P = Array(nS) { Array(nA) { mutableListOf<t4<Float, Int, Float, Boolean>>() } }
      val isd = f(nS) { if (desc[it / ncol][it % ncol] == 'S') 1.0f else 0.0f }
      fun to_s(row: Int, col: Int) = row * ncol + col
      
      fun inc(row: Int, col: Int, a: Int): t2<Int, Int> {
        return when (a) {
          0 -> t2(row, maxOf(col - 1, 0))
          1 -> t2(minOf(row + 1, nrow - 1), col)
          2 -> t2(row, minOf(col + 1, ncol - 1))
          3 -> t2(maxOf(row - 1, 0), col)
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
              li += t4(1.0f, s, 0.0f, true)
            else {
              if (is_slippery) {
                for (b in intArrayOf(((a - 1) % 4 + 4) % 4, a, (a + 1) % 4)) {
                  val (newrow, newcol) = inc(row, col, b)
                  val newstate = to_s(newrow, newcol)
                  val newletter = desc[newrow][newcol]
                  val done = newletter in "GH"
                  val rew = if (newletter == 'G') 1.0f else 0.0f
                  li += t4(1.0f / 3.0f, newstate, rew, done)
                }
              } else {
                val (newrow, newcol) = inc(row, col, a)
                val newstate = to_s(newrow, newcol)
                val newletter = desc[newrow][newcol]
                val done = newletter in "GH"
                val rew = if (newletter == 'G') 1.0f else 0.0f
                li += t4(1.0f, newstate, rew, done)
              }
            }
          }
        }
      return FrozenLake(desc, nrow, ncol, nS, nA, P, isd)
    }
  }
  
  override val reward_range = t2(0.0f, 1.0f)
  
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
}