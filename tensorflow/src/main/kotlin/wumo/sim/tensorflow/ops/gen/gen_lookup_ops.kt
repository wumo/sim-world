/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

object gen_lookup_ops {
  fun hashTable(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "HashTable"): Output =
      buildOpTensor("HashTable", name) {
        attr("key_dtype", keyDtype)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
      }
  
  fun hashTableV2(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "HashTableV2"): Output =
      buildOpTensor("HashTableV2", name) {
        attr("key_dtype", keyDtype)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
      }
  
  fun initializeTable(tableHandle: Output, keys: Output, values: Output, name: String = "InitializeTable"): Op =
      buildOp("InitializeTable", name) {
        addInput(tableHandle, true)
        addInput(keys, false)
        addInput(values, false)
      }
  
  fun initializeTableFromTextFile(tableHandle: Output, filename: Output, keyIndex: Long, valueIndex: Long, vocabSize: Long = -1L, delimiter: String = "\t", name: String = "InitializeTableFromTextFile"): Op =
      buildOp("InitializeTableFromTextFile", name) {
        addInput(tableHandle, true)
        addInput(filename, false)
        attr("key_index", keyIndex)
        attr("value_index", valueIndex)
        attr("vocab_size", vocabSize)
        attr("delimiter", delimiter)
      }
  
  fun initializeTableFromTextFileV2(tableHandle: Output, filename: Output, keyIndex: Long, valueIndex: Long, vocabSize: Long = -1L, delimiter: String = "\t", name: String = "InitializeTableFromTextFileV2"): Op =
      buildOp("InitializeTableFromTextFileV2", name) {
        addInput(tableHandle, false)
        addInput(filename, false)
        attr("key_index", keyIndex)
        attr("value_index", valueIndex)
        attr("vocab_size", vocabSize)
        attr("delimiter", delimiter)
      }
  
  fun initializeTableV2(tableHandle: Output, keys: Output, values: Output, name: String = "InitializeTableV2"): Op =
      buildOp("InitializeTableV2", name) {
        addInput(tableHandle, false)
        addInput(keys, false)
        addInput(values, false)
      }
  
  fun lookupTableExport(tableHandle: Output, tkeys: DataType<*>, tvalues: DataType<*>, name: String = "LookupTableExport"): List<Output> =
      buildOpTensors("LookupTableExport", name) {
        addInput(tableHandle, true)
        attr("Tkeys", tkeys)
        attr("Tvalues", tvalues)
      }
  
  fun lookupTableExportV2(tableHandle: Output, tkeys: DataType<*>, tvalues: DataType<*>, name: String = "LookupTableExportV2"): List<Output> =
      buildOpTensors("LookupTableExportV2", name) {
        addInput(tableHandle, false)
        attr("Tkeys", tkeys)
        attr("Tvalues", tvalues)
      }
  
  fun lookupTableFind(tableHandle: Output, keys: Output, defaultValue: Output, name: String = "LookupTableFind"): Output =
      buildOpTensor("LookupTableFind", name) {
        addInput(tableHandle, true)
        addInput(keys, false)
        addInput(defaultValue, false)
      }
  
  fun lookupTableFindV2(tableHandle: Output, keys: Output, defaultValue: Output, name: String = "LookupTableFindV2"): Output =
      buildOpTensor("LookupTableFindV2", name) {
        addInput(tableHandle, false)
        addInput(keys, false)
        addInput(defaultValue, false)
      }
  
  fun lookupTableImport(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableImport"): Op =
      buildOp("LookupTableImport", name) {
        addInput(tableHandle, true)
        addInput(keys, false)
        addInput(values, false)
      }
  
  fun lookupTableImportV2(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableImportV2"): Op =
      buildOp("LookupTableImportV2", name) {
        addInput(tableHandle, false)
        addInput(keys, false)
        addInput(values, false)
      }
  
  fun lookupTableInsert(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableInsert"): Op =
      buildOp("LookupTableInsert", name) {
        addInput(tableHandle, true)
        addInput(keys, false)
        addInput(values, false)
      }
  
  fun lookupTableInsertV2(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableInsertV2"): Op =
      buildOp("LookupTableInsertV2", name) {
        addInput(tableHandle, false)
        addInput(keys, false)
        addInput(values, false)
      }
  
  fun lookupTableSize(tableHandle: Output, name: String = "LookupTableSize"): Output =
      buildOpTensor("LookupTableSize", name) {
        addInput(tableHandle, true)
      }
  
  fun lookupTableSizeV2(tableHandle: Output, name: String = "LookupTableSizeV2"): Output =
      buildOpTensor("LookupTableSizeV2", name) {
        addInput(tableHandle, false)
      }
  
  fun mutableDenseHashTable(emptyKey: Output, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), initialNumBuckets: Long = 131072L, maxLoadFactor: Float = 0.8f, name: String = "MutableDenseHashTable"): Output =
      buildOpTensor("MutableDenseHashTable", name) {
        addInput(emptyKey, false)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
        attr("value_shape", valueShape)
        attr("initial_num_buckets", initialNumBuckets)
        attr("max_load_factor", maxLoadFactor)
      }
  
  fun mutableDenseHashTableV2(emptyKey: Output, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), initialNumBuckets: Long = 131072L, maxLoadFactor: Float = 0.8f, name: String = "MutableDenseHashTableV2"): Output =
      buildOpTensor("MutableDenseHashTableV2", name) {
        addInput(emptyKey, false)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
        attr("value_shape", valueShape)
        attr("initial_num_buckets", initialNumBuckets)
        attr("max_load_factor", maxLoadFactor)
      }
  
  fun mutableHashTable(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "MutableHashTable"): Output =
      buildOpTensor("MutableHashTable", name) {
        attr("key_dtype", keyDtype)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
      }
  
  fun mutableHashTableOfTensors(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), name: String = "MutableHashTableOfTensors"): Output =
      buildOpTensor("MutableHashTableOfTensors", name) {
        attr("key_dtype", keyDtype)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
        attr("value_shape", valueShape)
      }
  
  fun mutableHashTableOfTensorsV2(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), name: String = "MutableHashTableOfTensorsV2"): Output =
      buildOpTensor("MutableHashTableOfTensorsV2", name) {
        attr("key_dtype", keyDtype)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
        attr("value_shape", valueShape)
      }
  
  fun mutableHashTableV2(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "MutableHashTableV2"): Output =
      buildOpTensor("MutableHashTableV2", name) {
        attr("key_dtype", keyDtype)
        attr("value_dtype", valueDtype)
        attr("container", container)
        attr("shared_name", sharedName)
        attr("use_node_name_sharing", useNodeNameSharing)
      }
}