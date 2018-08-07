/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.algorithm.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.algorithm.tensorflow.buildOpTensor
import wumo.sim.algorithm.tensorflow.buildOpTensors
import wumo.sim.util.Shape

fun TF.decodeCSV(records: Output, record_defaults: Output, field_delim: String = ",", use_quote_delim: Boolean = true, na_value: String = "", select_cols: Array<Long> = arrayOf(), name: String = "DecodeCSV") = run {
  buildOpTensors("DecodeCSV", name) {
    addInput(records, false)
    addInput(record_defaults, false)
    attr("field_delim", field_delim)
    attr("use_quote_delim", use_quote_delim)
    attr("na_value", na_value)
    attr("select_cols", select_cols)
  }
}

fun TF.decodeCompressed(bytes: Output, compression_type: String = "", name: String = "DecodeCompressed") = run {
  buildOpTensor("DecodeCompressed", name) {
    addInput(bytes, false)
    attr("compression_type", compression_type)
  }
}

fun TF.decodeJSONExample(json_examples: Output, name: String = "DecodeJSONExample") = run {
  buildOpTensor("DecodeJSONExample", name) {
    addInput(json_examples, false)
  }
}

fun TF.decodeRaw(bytes: Output, out_type: Int, little_endian: Boolean = true, name: String = "DecodeRaw") = run {
  buildOpTensor("DecodeRaw", name) {
    addInput(bytes, false)
    attrType("out_type", out_type)
    attr("little_endian", little_endian)
  }
}

fun TF.parseExample(serialized: Output, names: Output, sparse_keys: Array<Output>, dense_keys: Array<Output>, dense_defaults: Output, sparse_types: Array<Long>, dense_shapes: Array<Shape>, name: String = "ParseExample") = run {
  buildOpTensors("ParseExample", name) {
    addInput(serialized, false)
    addInput(names, false)
    addInput(sparse_keys, false)
    addInput(dense_keys, false)
    addInput(dense_defaults, false)
    attr("sparse_types", sparse_types)
    attr("dense_shapes", dense_shapes)
  }
}

fun TF.parseSingleExample(serialized: Output, dense_defaults: Output, num_sparse: Long, sparse_keys: Array<String>, dense_keys: Array<String>, sparse_types: Array<Long>, dense_shapes: Array<Shape>, name: String = "ParseSingleExample") = run {
  buildOpTensors("ParseSingleExample", name) {
    addInput(serialized, false)
    addInput(dense_defaults, false)
    attr("num_sparse", num_sparse)
    attr("sparse_keys", sparse_keys)
    attr("dense_keys", dense_keys)
    attr("sparse_types", sparse_types)
    attr("dense_shapes", dense_shapes)
  }
}

fun TF.parseSingleSequenceExample(serialized: Output, feature_list_dense_missing_assumed_empty: Output, context_sparse_keys: Array<Output>, context_dense_keys: Array<Output>, feature_list_sparse_keys: Array<Output>, feature_list_dense_keys: Array<Output>, context_dense_defaults: Output, debug_name: Output, context_sparse_types: Array<Long> = arrayOf(), feature_list_dense_types: Array<Long> = arrayOf(), context_dense_shapes: Array<Shape> = arrayOf(), feature_list_sparse_types: Array<Long> = arrayOf(), feature_list_dense_shapes: Array<Shape> = arrayOf(), name: String = "ParseSingleSequenceExample") = run {
  buildOpTensors("ParseSingleSequenceExample", name) {
    addInput(serialized, false)
    addInput(feature_list_dense_missing_assumed_empty, false)
    addInput(context_sparse_keys, false)
    addInput(context_dense_keys, false)
    addInput(feature_list_sparse_keys, false)
    addInput(feature_list_dense_keys, false)
    addInput(context_dense_defaults, false)
    addInput(debug_name, false)
    attr("context_sparse_types", context_sparse_types)
    attr("feature_list_dense_types", feature_list_dense_types)
    attr("context_dense_shapes", context_dense_shapes)
    attr("feature_list_sparse_types", feature_list_sparse_types)
    attr("feature_list_dense_shapes", feature_list_dense_shapes)
  }
}

fun TF.parseTensor(serialized: Output, out_type: Int, name: String = "ParseTensor") = run {
  buildOpTensor("ParseTensor", name) {
    addInput(serialized, false)
    attrType("out_type", out_type)
  }
}

fun TF.serializeTensor(tensor: Output, name: String = "SerializeTensor") = run {
  buildOpTensor("SerializeTensor", name) {
    addInput(tensor, false)
  }
}

fun TF.stringToNumber(string_tensor: Output, out_type: Int = DT_FLOAT, name: String = "StringToNumber") = run {
  buildOpTensor("StringToNumber", name) {
    addInput(string_tensor, false)
    attrType("out_type", out_type)
  }
}
