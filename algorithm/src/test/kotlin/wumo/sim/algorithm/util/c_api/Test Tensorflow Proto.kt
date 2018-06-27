package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.junit.Test
import org.tensorflow.framework.AttrValue
import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorProto
import org.tensorflow.framework.TensorShapeProto

class `Test Tensorflow Proto` {
  @Test
  fun `test Java proto`() {
    val builder = TensorProto.newBuilder()
    with(builder) {
      dtype = DataType.DT_FLOAT
      tensorShape = TensorShapeProto.newBuilder().apply {
        addDim(TensorShapeProto.Dim.newBuilder().setSize(16))
        addDim(TensorShapeProto.Dim.newBuilder().setSize(4))
      }.build()
      addFloatVal(9f)
    }
    
    val tensorProto = builder.build()
    println(tensorProto)
    println(tensorProto.serializedSize)
    val bytes = tensorProto.toByteArray()
    println(bytes.size)
    println(String(bytes))
    
    AttrValue.newBuilder().apply {
      tensor = tensorProto
      val attrValue = build()
      println(attrValue)
    }
  }
  
  @Test
  fun `test c api proto`() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
    
    val tensorProto = tensorflow.TensorProto()
    tensorProto.set_dtype(DT_FLOAT)
    tensorProto.mutable_tensor_shape().apply {
      add_dim().set_size(16)
      add_dim().set_size(4)
    }
    tensorProto.add_float_val(9f)
    val s = tensorProto.SerializeAsString()
    val t = TensorProto.parseFrom(s.asByteBuffer())
    println(t)
    val attrValue = tensorflow.AttrValue()
    attrValue.set_allocated_tensor(tensorProto)
    val a = attrValue.SerializeAsString()
    
    val len = attrValue.ByteSizeLong()
    println("limit: " + a.limit())
    println("len: " + len)
    val aa = AttrValue.parseFrom(a.asByteBuffer())
    println(aa)
  }
}