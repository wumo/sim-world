package wumo.sim.algorithm.tensorflow.learn_lowlevel_api

import org.junit.Test
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.base_dtype
import wumo.sim.algorithm.tensorflow.ops.*
import wumo.sim.algorithm.tensorflow.tf

class variable_initialization : BaseTest() {
  @Test
  fun `variable depend on variable`() {
    val initial_value = tf.const(2f, "initial_value")
    val v = tf.g.nodeBuilder("VariableV2", "v")
        .attrType("dtype", initial_value.dtype.base_dtype)
        .attr("shape", initial_value.shape)
        .build()
    
    val vt = Tensor(v, 0)
    val v_init = tf.g.nodeBuilder("Assign", "v/init")
        .addInput(vt)
        .addInput(initial_value)
        .build()
    val v_read = tf.identity(vt, name = "v/read")
    
    val w = tf.g.nodeBuilder("VariableV2", "w")
        .attrType("dtype", vt.dtype.base_dtype)
        .attr("shape", vt.shape)
        .build()
    
    val wt = Tensor(w, 0)
    
    val v_is_initialized = tf.is_variable_initialized(vt, "v_is_initialized")
    val v_initialized = tf.ref_switch(vt, v_is_initialized, "switch_v_initialized")[1]
    val v_not_initialized = tf.switch(initial_value, v_is_initialized, "switch_v_not_initialized")[0]
    val v_initialized_value = tf.merge(v_initialized, v_not_initialized, name = "v_initialized_value")[0]
    
    val w_init = tf.g.nodeBuilder("Assign", "w/init")
        .addInput(wt)
        .addInput(v_initialized_value)
        .build()
    
    printGraph()
    tf.session {
      w_init.run()
      v_init.run()
      vt.eval()
      wt.eval()
    }
  }
}