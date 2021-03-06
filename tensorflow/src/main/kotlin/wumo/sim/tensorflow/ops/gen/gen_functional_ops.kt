/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.NameAttrList
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

object gen_functional_ops {
  fun fakeParam(dtype: DataType<*>, shape: Shape, name: String = "FakeParam"): Output =
      buildOpTensor("FakeParam", name) {
        attr("dtype", dtype)
        attr("shape", shape)
      }
  
  fun _for(start: Output, limit: Output, delta: Output, input: Output, body: NameAttrList, name: String = "For"): List<Output> =
      buildOpTensors("For", name) {
        addInput(start, false)
        addInput(limit, false)
        addInput(delta, false)
        addInput(input, false)
        attr("body", body)
      }
  
  fun _if(cond: Output, input: Output, tout: Array<Long>, thenBranch: NameAttrList, elseBranch: NameAttrList, name: String = "If"): List<Output> =
      buildOpTensors("If", name) {
        addInput(cond, false)
        addInput(input, false)
        attr("Tout", tout)
        attr("then_branch", thenBranch)
        attr("else_branch", elseBranch)
      }
  
  fun partitionedCall(args: Output, tout: Array<Long>, f: NameAttrList, name: String = "PartitionedCall"): List<Output> =
      buildOpTensors("PartitionedCall", name) {
        addInput(args, false)
        attr("Tout", tout)
        attr("f", f)
      }
  
  fun remoteCall(target: Output, args: Output, tout: Array<Long>, f: NameAttrList, name: String = "RemoteCall"): List<Output> =
      buildOpTensors("RemoteCall", name) {
        addInput(target, false)
        addInput(args, false)
        attr("Tout", tout)
        attr("f", f)
      }
  
  fun statefulPartitionedCall(args: Output, tout: Array<Long>, f: NameAttrList, name: String = "StatefulPartitionedCall"): List<Output> =
      buildOpTensors("StatefulPartitionedCall", name) {
        addInput(args, false)
        attr("Tout", tout)
        attr("f", f)
      }
  
  fun symbolicGradient(input: Output, tout: Array<Long>, f: NameAttrList, name: String = "SymbolicGradient"): List<Output> =
      buildOpTensors("SymbolicGradient", name) {
        addInput(input, false)
        attr("Tout", tout)
        attr("f", f)
      }
  
  fun _while(input: Output, cond: NameAttrList, body: NameAttrList, name: String = "While"): List<Output> =
      buildOpTensors("While", name) {
        addInput(input, false)
        attr("cond", cond)
        attr("body", body)
      }
}