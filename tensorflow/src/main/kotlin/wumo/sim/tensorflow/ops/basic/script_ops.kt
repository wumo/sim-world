package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_script_ops

object script_ops {
  interface API {
    fun eagerPyFunc(input: Output, token: String, tout: Array<Long>, name: String = "EagerPyFunc"): List<Output> {
      return gen_script_ops.eagerPyFunc(input, token, tout, name)
    }
    
    fun pyFunc(input: Output, token: String, tout: Array<Long>, name: String = "PyFunc"): List<Output> {
      return gen_script_ops.pyFunc(input, token, tout, name)
    }
    
    fun pyFuncStateless(input: Output, token: String, tout: Array<Long>, name: String = "PyFuncStateless"): List<Output> {
      return gen_script_ops.pyFuncStateless(input, token, tout, name)
    }
  }
}