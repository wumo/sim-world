package wumo.sim.util.ndarray

import org.junit.Test

import org.junit.Assert.*
import wumo.sim.util.f
import wumo.sim.util.x

class MathKtTest {
  
  @Test
  fun ones_like() {
  }
  
  @Test
  fun test_newaxis() {
    val a = NDArray(2 x 2, 1f)
    val b = newaxis(a)
    println("a=\n$a")
    println("b=\n$b")
  }
}