package wumo.sim.util.ndarray

import org.junit.Test
import wumo.sim.util.Shape

class MathKtTest {
  
  @Test
  fun ones_like() {
  }
  
  @Test
  fun test_newaxis() {
    val a = NDArray(Shape(2, 2), 1f)
    val b = newaxis(a)
    println("a=\n$a")
    println("b=\n$b")
  }
  
  @Test
  fun divAssign() {
    val a = NDArray(Shape(2, 2), 1f)
    println(a)
    a /= 2f
    println(a)
  }
}