package wumo.sim.algorithm.tensorflow.scope

import org.junit.Assert.*
import org.junit.Test
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.ops.get_variable
import wumo.sim.algorithm.tensorflow.ops.variable

import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.f

class VariableScopeTest {
  
  @Test
  fun subscope() {
    tf.name_scope("name_scope_x") {
      val v1 = tf.get_variable(tf.const(1), name = "var1")
      assertEquals("var1", v1.name)
      val v2 = tf.variable(2, name = "var2")
      tf.variable_scope("variable_scope_y") {
        val v1 = tf.get_variable(tf.const(1), name = "var1")
        tf.ctxVs.reuse = true
        val v1_reuse = tf.get_variable(name = "var1")
        val var2 = tf.variable(f(2f), name = "var2")
        val var2_reuse = tf.variable(f(2f), name = "var2")
        println()
      }
    }
    val init = tf.global_variable_initializer()
    tf.printGraph()
  }
  
  @Test
  fun `variable scope test 2`() {
    tf.name_scope("L1") {
      val d = tf.const(1, name = "c")
      tf.name_scope("L2") {
        val a = tf.const(1, name = "a")
        tf.variable_scope("L1") {
          val c = tf.get_variable(f(1f), "c")
          val g = tf.const(1, name = "g")
          assertEquals("L1/L2/a", a.name)
          assertEquals("L1/c_1", c.name)
          assertEquals("L1/L2/L1/g", g.name)
        }
      }
    }
    
  }
}