package wumo.sim.tensorflow.learn_lowlevel_api

import org.junit.Test
import wumo.sim.tensorflow.ops.BaseTest
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.tf

class variable_initialization : BaseTest() {
  @Test
  fun `variable depend on variable`() {
    val initial_value = tf.const(2f, "initial_value")
    val v = tf.currentGraph.nodeBuilder("VariableV2", "v").run {
      attr("dtype", initial_value.dtype.base_dtype)
      attr("shape", initial_value.shape)
      build()
    }
    
    val vt = Output(v, 0)
    val v_init = tf.currentGraph.nodeBuilder("Assign", "v/init").run {
      addInput(vt)
      addInput(initial_value)
      build()
    }
    val v_read = tf.identity(vt, name = "v/read")
    
    val w = tf.currentGraph.nodeBuilder("VariableV2", "w").run {
      attr("dtype", vt.dtype.base_dtype)
      attr("shape", vt.shape)
      build()
    }
    
    val wt = Output(w, 0)
    
    val v_is_initialized = tf._isVariableInitialized(vt, "v_is_initialized")
    val v_initialized = tf._refSwitch(vt, v_is_initialized, "switch_v_initialized")[1]
    val v_not_initialized = tf._switch(initial_value, v_is_initialized, "switch_v_not_initialized")[0]
//    val v_initialized_value = tf._merge(v_initialized, v_not_initialized, name = "v_initialized_value")[0]
    
    val w_init = tf.currentGraph.nodeBuilder("Assign", "w/init").run {
      addInput(wt)
//      addInput(v_initialized_value)
      build()
    }
    
    printGraph()
    tf.session {
      w_init.run()
      v_init.run()
      vt.eval()
      wt.eval()
    }
  }
}