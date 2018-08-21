package wumo.sim.tensorflow.scope

import org.junit.Assert.assertEquals
import org.junit.Test
import wumo.sim.tensorflow.tf
import wumo.sim.util.f

class VariableScopeTest {
  
  @Test
  fun subscope() {
//    tf.nameScope("name_scope_x") {
//      val v1 = tf.get_variable(tf.const(1), name = "var1")
//      assertEquals("var1", v1.name)
//      val v2 = tf.variable(2, name = "var2")
//      tf.variable_scope("variable_scope_y") {
//        val v1 = tf.get_variable(tf.const(1), name = "var1")
//        tf.ctxVs.reuse = true
//        val v1_reuse = tf.get_variable(name = "var1")
//        val var2 = tf.variable(f(2f), name = "var2")
//        val var2_reuse = tf.variable(f(2f), name = "var2")
//        println()
//      }
//    }
//    val init = tf.globalVariablesInitializer()
//    tf.printGraph()
  }
  
  @Test
  fun `variable scope test 2`() {
    tf.nameScope("L1") {
      val d = tf.const(1, name = "c")
      tf.nameScope("L2") {
        val a = tf.const(1, name = "a")
        tf.variableScope("L1") {
          val c = tf.variable(f(1f), "c")
          val g = tf.const(1, name = "g")
          assertEquals("L1/L2/a:0", a.name)
          assertEquals("L1/c_1", c.name)
          assertEquals("L1/L2/L1/g:0", g.name)
        }
      }
    }
    
  }
}