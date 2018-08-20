package wumo.sim.util

import org.junit.Test

class ControlFlowTest {
  @Test
  fun `switch type 2 test`() {
    val switch = SwitchType2<Int, Double>().apply {
      case<String> { println("$_1,$_2");1.0 }
      case<Int> { println("$_1,$_2");2.0 }
      case<Float> { println("$_1,$_2");2.0 }
    }
    
    val a = switch("string", 1)
    println(a)
    val b = switch(1, 2)
    println(b)
    val c = switch(2f, 3)
    println(c)
  }
  
  @Test
  fun `switch return type test`() {
    val switch = SwitchReturnType().apply {
      case<String, Int> { it.toInt() }
      case<String, String> { it }
      case<Float, String> { it.toString() }
      case<Double, String> { it.toString() }
    }
    
    val a = switch<String>(1.0)
    println(a)
    val b = switch<Int>("2")
    println(b)
  }
  
  interface A<T, R> : ReturnType<T, R> {
    fun print()
  }
  
  @Test
  fun `switch return type class test`() {
    
    val switch = SwitchReturnTypeClass().apply {
      case(object : A<String, Int> {
        override fun print() {
          println("string,int")
        }
      })
      case(object : A<String, String> {
        override fun print() {
          println("string,String")
        }
      })
      case(object : A<Double, String> {
        override fun print() {
          println("double,String")
        }
      })
    }
    
    val a = switch<String, Double>(1.0)
    (a as A<String,Double>).print()
  }
}