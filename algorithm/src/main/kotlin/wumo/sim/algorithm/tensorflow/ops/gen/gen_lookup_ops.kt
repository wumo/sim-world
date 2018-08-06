/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.algorithm.tensorflow.ops.gen

import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.util.Dimension

fun TF.hashTableV2(key_dtype: Int, value_dtype: Int, container: String = "", shared_name: String = "", use_node_name_sharing: Boolean = false, name: String = "HashTableV2") = run {
  buildOpTensor("HashTableV2", name) {
    attrType("key_dtype", key_dtype)
    attrType("value_dtype", value_dtype)
    attr("container", container)
    attr("shared_name", shared_name)
    attr("use_node_name_sharing", use_node_name_sharing)
  }
}

fun TF.initializeTableFromTextFileV2(table_handle: Output, filename: Output, key_index: Long, value_index: Long, vocab_size: Long = -1L, delimiter: String = "\t", name: String = "InitializeTableFromTextFileV2") = run {
  buildOp("InitializeTableFromTextFileV2", name) {
    addInput(table_handle, false)
    addInput(filename, false)
    attr("key_index", key_index)
    attr("value_index", value_index)
    attr("vocab_size", vocab_size)
    attr("delimiter", delimiter)
  }
}

fun TF.initializeTableV2(table_handle: Output, keys: Output, values: Output, name: String = "InitializeTableV2") = run {
  buildOp("InitializeTableV2", name) {
    addInput(table_handle, false)
    addInput(keys, false)
    addInput(values, false)
  }
}

fun TF.lookupTableExportV2(table_handle: Output, tkeys: Int, tvalues: Int, name: String = "LookupTableExportV2") = run {
  buildOpTensors("LookupTableExportV2", name) {
    addInput(table_handle, false)
    attrType("Tkeys", tkeys)
    attrType("Tvalues", tvalues)
  }
}

fun TF.lookupTableFindV2(table_handle: Output, keys: Output, default_value: Output, name: String = "LookupTableFindV2") = run {
  buildOpTensor("LookupTableFindV2", name) {
    addInput(table_handle, false)
    addInput(keys, false)
    addInput(default_value, false)
  }
}

fun TF.lookupTableImportV2(table_handle: Output, keys: Output, values: Output, name: String = "LookupTableImportV2") = run {
  buildOp("LookupTableImportV2", name) {
    addInput(table_handle, false)
    addInput(keys, false)
    addInput(values, false)
  }
}

fun TF.lookupTableInsertV2(table_handle: Output, keys: Output, values: Output, name: String = "LookupTableInsertV2") = run {
  buildOp("LookupTableInsertV2", name) {
    addInput(table_handle, false)
    addInput(keys, false)
    addInput(values, false)
  }
}

fun TF.lookupTableSizeV2(table_handle: Output, name: String = "LookupTableSizeV2") = run {
  buildOpTensor("LookupTableSizeV2", name) {
    addInput(table_handle, false)
  }
}

fun TF.mutableDenseHashTableV2(empty_key: Output, value_dtype: Int, container: String = "", shared_name: String = "", use_node_name_sharing: Boolean = false, value_shape: Dimension = Dimension(longArrayOf()), initial_num_buckets: Long = 131072L, max_load_factor: Float = 0.8f, name: String = "MutableDenseHashTableV2") = run {
  buildOpTensor("MutableDenseHashTableV2", name) {
    addInput(empty_key, false)
    attrType("value_dtype", value_dtype)
    attr("container", container)
    attr("shared_name", shared_name)
    attr("use_node_name_sharing", use_node_name_sharing)
    attr("value_shape", value_shape)
    attr("initial_num_buckets", initial_num_buckets)
    attr("max_load_factor", max_load_factor)
  }
}

fun TF.mutableHashTableOfTensorsV2(key_dtype: Int, value_dtype: Int, container: String = "", shared_name: String = "", use_node_name_sharing: Boolean = false, value_shape: Dimension = Dimension(longArrayOf()), name: String = "MutableHashTableOfTensorsV2") = run {
  buildOpTensor("MutableHashTableOfTensorsV2", name) {
    attrType("key_dtype", key_dtype)
    attrType("value_dtype", value_dtype)
    attr("container", container)
    attr("shared_name", shared_name)
    attr("use_node_name_sharing", use_node_name_sharing)
    attr("value_shape", value_shape)
  }
}

fun TF.mutableHashTableV2(key_dtype: Int, value_dtype: Int, container: String = "", shared_name: String = "", use_node_name_sharing: Boolean = false, name: String = "MutableHashTableV2") = run {
  buildOpTensor("MutableHashTableV2", name) {
    attrType("key_dtype", key_dtype)
    attrType("value_dtype", value_dtype)
    attr("container", container)
    attr("shared_name", shared_name)
    attr("use_node_name_sharing", use_node_name_sharing)
  }
}
