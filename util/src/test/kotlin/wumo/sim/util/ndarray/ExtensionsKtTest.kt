package wumo.sim.util.ndarray

import org.junit.Test

import wumo.sim.util.Shape

class ExtensionsKtTest {
  
  @Test
  fun testConcatenate() {
    val a = NDArray(Shape(2, 2), intArrayOf(1, 2, 3, 4))
    var b = NDArray(Shape(1, 2), intArrayOf(5, 6))
    var c = concatenate(listOf(a, b), axis = 0)
    println(a)
    println(b)
    println(c)
    b = b.reshape(Shape(2, 1))
    c = concatenate(listOf(a, b), axis = 1)
    println(a)
    println(b)
    println(c)
  }
}