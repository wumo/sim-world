package wumo.sim.tensorflow.learn_lowlevel_api

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.tensorflow.TF_GetAllOpList
import org.junit.Test
import org.tensorflow.framework.OpList
import wumo.sim.algorithm.tensorflow.TF

class protobuf_test {
  @Test
  fun use_protobuf_api() {
    TF
    val opdef = TF_GetAllOpList()
    val data = opdef.data()
    data.limit<Pointer>(opdef.length())
    val oplist = OpList.parseFrom(data.asByteBuffer()).opList
    val opDefsMap = oplist.map { it.name to it }.toMap()
    
  }
}