package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_lookup_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

object lookup_ops {
  interface API {
    fun hashTable(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "HashTable"): Output {
      return gen_lookup_ops.hashTable(keyDtype, valueDtype, container, sharedName, useNodeNameSharing, name)
    }
    
    fun hashTableV2(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "HashTableV2"): Output {
      return gen_lookup_ops.hashTableV2(keyDtype, valueDtype, container, sharedName, useNodeNameSharing, name)
    }
    
    fun initializeTable(tableHandle: Output, keys: Output, values: Output, name: String = "InitializeTable"): Op {
      return gen_lookup_ops.initializeTable(tableHandle, keys, values, name)
    }
    
    fun initializeTableFromTextFile(tableHandle: Output, filename: Output, keyIndex: Long, valueIndex: Long, vocabSize: Long = -1L, delimiter: String = "\t", name: String = "InitializeTableFromTextFile"): Op {
      return gen_lookup_ops.initializeTableFromTextFile(tableHandle, filename, keyIndex, valueIndex, vocabSize, delimiter, name)
    }
    
    fun initializeTableFromTextFileV2(tableHandle: Output, filename: Output, keyIndex: Long, valueIndex: Long, vocabSize: Long = -1L, delimiter: String = "\t", name: String = "InitializeTableFromTextFileV2"): Op {
      return gen_lookup_ops.initializeTableFromTextFileV2(tableHandle, filename, keyIndex, valueIndex, vocabSize, delimiter, name)
    }
    
    fun initializeTableV2(tableHandle: Output, keys: Output, values: Output, name: String = "InitializeTableV2"): Op {
      return gen_lookup_ops.initializeTableV2(tableHandle, keys, values, name)
    }
    
    fun lookupTableExport(tableHandle: Output, tkeys: DataType<*>, tvalues: DataType<*>, name: String = "LookupTableExport"): List<Output> {
      return gen_lookup_ops.lookupTableExport(tableHandle, tkeys, tvalues, name)
    }
    
    fun lookupTableExportV2(tableHandle: Output, tkeys: DataType<*>, tvalues: DataType<*>, name: String = "LookupTableExportV2"): List<Output> {
      return gen_lookup_ops.lookupTableExportV2(tableHandle, tkeys, tvalues, name)
    }
    
    fun lookupTableFind(tableHandle: Output, keys: Output, defaultValue: Output, name: String = "LookupTableFind"): Output {
      return gen_lookup_ops.lookupTableFind(tableHandle, keys, defaultValue, name)
    }
    
    fun lookupTableFindV2(tableHandle: Output, keys: Output, defaultValue: Output, name: String = "LookupTableFindV2"): Output {
      return gen_lookup_ops.lookupTableFindV2(tableHandle, keys, defaultValue, name)
    }
    
    fun lookupTableImport(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableImport"): Op {
      return gen_lookup_ops.lookupTableImport(tableHandle, keys, values, name)
    }
    
    fun lookupTableImportV2(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableImportV2"): Op {
      return gen_lookup_ops.lookupTableImportV2(tableHandle, keys, values, name)
    }
    
    fun lookupTableInsert(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableInsert"): Op {
      return gen_lookup_ops.lookupTableInsert(tableHandle, keys, values, name)
    }
    
    fun lookupTableInsertV2(tableHandle: Output, keys: Output, values: Output, name: String = "LookupTableInsertV2"): Op {
      return gen_lookup_ops.lookupTableInsertV2(tableHandle, keys, values, name)
    }
    
    fun lookupTableSize(tableHandle: Output, name: String = "LookupTableSize"): Output {
      return gen_lookup_ops.lookupTableSize(tableHandle, name)
    }
    
    fun lookupTableSizeV2(tableHandle: Output, name: String = "LookupTableSizeV2"): Output {
      return gen_lookup_ops.lookupTableSizeV2(tableHandle, name)
    }
    
    fun mutableDenseHashTable(emptyKey: Output, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), initialNumBuckets: Long = 131072L, maxLoadFactor: Float = 0.8f, name: String = "MutableDenseHashTable"): Output {
      return gen_lookup_ops.mutableDenseHashTable(emptyKey, valueDtype, container, sharedName, useNodeNameSharing, valueShape, initialNumBuckets, maxLoadFactor, name)
    }
    
    fun mutableDenseHashTableV2(emptyKey: Output, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), initialNumBuckets: Long = 131072L, maxLoadFactor: Float = 0.8f, name: String = "MutableDenseHashTableV2"): Output {
      return gen_lookup_ops.mutableDenseHashTableV2(emptyKey, valueDtype, container, sharedName, useNodeNameSharing, valueShape, initialNumBuckets, maxLoadFactor, name)
    }
    
    fun mutableHashTable(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "MutableHashTable"): Output {
      return gen_lookup_ops.mutableHashTable(keyDtype, valueDtype, container, sharedName, useNodeNameSharing, name)
    }
    
    fun mutableHashTableOfTensors(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), name: String = "MutableHashTableOfTensors"): Output {
      return gen_lookup_ops.mutableHashTableOfTensors(keyDtype, valueDtype, container, sharedName, useNodeNameSharing, valueShape, name)
    }
    
    fun mutableHashTableOfTensorsV2(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, valueShape: Shape = Shape(longArrayOf()), name: String = "MutableHashTableOfTensorsV2"): Output {
      return gen_lookup_ops.mutableHashTableOfTensorsV2(keyDtype, valueDtype, container, sharedName, useNodeNameSharing, valueShape, name)
    }
    
    fun mutableHashTableV2(keyDtype: DataType<*>, valueDtype: DataType<*>, container: String = "", sharedName: String = "", useNodeNameSharing: Boolean = false, name: String = "MutableHashTableV2"): Output {
      return gen_lookup_ops.mutableHashTableV2(keyDtype, valueDtype, container, sharedName, useNodeNameSharing, name)
    }
  }
}