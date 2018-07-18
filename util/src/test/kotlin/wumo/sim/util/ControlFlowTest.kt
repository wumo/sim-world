package wumo.sim.util

import org.junit.Assert.*
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
}