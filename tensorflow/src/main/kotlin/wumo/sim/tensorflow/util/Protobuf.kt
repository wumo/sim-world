package wumo.sim.tensorflow.util

import org.bytedeco.javacpp.tensorflow

fun attrValue(value: Boolean): tensorflow.AttrValue {
  val attr = tensorflow.AttrValue()
  attr.set_b(value)
  return attr
}

fun attrValue(value: String): tensorflow.AttrValue {
  val attr = tensorflow.AttrValue()
  attr.set_s(value)
  return attr
}