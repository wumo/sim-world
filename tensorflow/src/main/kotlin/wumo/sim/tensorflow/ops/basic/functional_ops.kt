package wumo.sim.tensorflow.ops.basic

import org.bytedeco.javacpp.tensorflow.NameAttrList
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_functional_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

object functional_ops {
  interface API {
    fun fakeParam(dtype: DataType<*>, shape: Shape, name: String = "FakeParam"): Output {
      return gen_functional_ops.fakeParam(dtype, shape, name)
    }
    
    fun _for(start: Output, limit: Output, delta: Output, input: Output, body: NameAttrList, name: String = "For"): List<Output> {
      return gen_functional_ops._for(start, limit, delta, input, body, name)
    }
    
    fun _if(cond: Output, input: Output, tout: Array<Long>, thenBranch: NameAttrList, elseBranch: NameAttrList, name: String = "If"): List<Output> {
      return gen_functional_ops._if(cond, input, tout, thenBranch, elseBranch, name)
    }
    
    fun partitionedCall(args: Output, tout: Array<Long>, f: NameAttrList, name: String = "PartitionedCall"): List<Output> {
      return gen_functional_ops.partitionedCall(args, tout, f, name)
    }
    
    fun remoteCall(target: Output, args: Output, tout: Array<Long>, f: NameAttrList, name: String = "RemoteCall"): List<Output> {
      return gen_functional_ops.remoteCall(target, args, tout, f, name)
    }
    
    fun statefulPartitionedCall(args: Output, tout: Array<Long>, f: NameAttrList, name: String = "StatefulPartitionedCall"): List<Output> {
      return gen_functional_ops.statefulPartitionedCall(args, tout, f, name)
    }
    
    fun symbolicGradient(input: Output, tout: Array<Long>, f: NameAttrList, name: String = "SymbolicGradient"): List<Output> {
      return gen_functional_ops.symbolicGradient(input, tout, f, name)
    }
    
    fun _while(input: Output, cond: NameAttrList, body: NameAttrList, name: String = "While"): List<Output> {
      return gen_functional_ops._while(input, cond, body, name)
    }
  }
}