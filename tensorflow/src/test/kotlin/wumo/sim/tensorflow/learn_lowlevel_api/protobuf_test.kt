package wumo.sim.tensorflow.learn_lowlevel_api

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Buffer.newBuffer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.TF_GetAllOpList
import org.bytedeco.javacpp.tensorflow.TF_GraphGetOpDef
import org.junit.Test
import org.tensorflow.framework.OpDef
import org.tensorflow.framework.OpList
import wumo.sim.tensorflow.tf

class protobuf_test {
  @Test
  fun use_protobuf_api() {
    tf
    val opdef = TF_GetAllOpList()
    val data = opdef.data()
    data.limit<Pointer>(opdef.length())
    val oplist = OpList.parseFrom(data.asByteBuffer()).opList
    val opDefsMap = oplist.map { it.name to it }.toMap()
  }
  
  @Test
  fun get_op_def() {
    tf
    val buf = newBuffer()
    val status = newStatus()
    TF_GraphGetOpDef(tf.currentGraph.c_graph, "Merge", buf, status)
    val data = buf.data()
    
    data.limit<Pointer>(buf.length())
    val op_def = OpDef.parseFrom(data.asByteBuffer())
    println(op_def)
  }
}