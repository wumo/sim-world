package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_parsing_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.util.Shape

object parsing_ops {
  interface API {
    fun decodeCSV(records: Output, recordDefaults: Output, fieldDelim: String = ",", useQuoteDelim: Boolean = true, naValue: String = "", selectCols: Array<Long> = arrayOf(), name: String = "DecodeCSV"): List<Output> {
      return gen_parsing_ops.decodeCSV(records, recordDefaults, fieldDelim, useQuoteDelim, naValue, selectCols, name)
    }
    
    fun decodeCompressed(bytes: Output, compressionType: String = "", name: String = "DecodeCompressed"): Output {
      return gen_parsing_ops.decodeCompressed(bytes, compressionType, name)
    }
    
    fun decodeJSONExample(jsonExamples: Output, name: String = "DecodeJSONExample"): Output {
      return gen_parsing_ops.decodeJSONExample(jsonExamples, name)
    }
    
    fun decodeRaw(bytes: Output, outType: DataType<*>, littleEndian: Boolean = true, name: String = "DecodeRaw"): Output {
      return gen_parsing_ops.decodeRaw(bytes, outType, littleEndian, name)
    }
    
    fun parseExample(serialized: Output, names: Output, sparseKeys: List<Output>, denseKeys: List<Output>, denseDefaults: Output, sparseTypes: Array<Long>, denseShapes: Array<Shape>, name: String = "ParseExample"): List<Output> {
      return gen_parsing_ops.parseExample(serialized, names, sparseKeys, denseKeys, denseDefaults, sparseTypes, denseShapes, name)
    }
    
    fun parseSingleExample(serialized: Output, denseDefaults: Output, numSparse: Long, sparseKeys: Array<String>, denseKeys: Array<String>, sparseTypes: Array<Long>, denseShapes: Array<Shape>, name: String = "ParseSingleExample"): List<Output> {
      return gen_parsing_ops.parseSingleExample(serialized, denseDefaults, numSparse, sparseKeys, denseKeys, sparseTypes, denseShapes, name)
    }
    
    fun parseSingleSequenceExample(serialized: Output, featureListDenseMissingAssumedEmpty: Output, contextSparseKeys: List<Output>, contextDenseKeys: List<Output>, featureListSparseKeys: List<Output>, featureListDenseKeys: List<Output>, contextDenseDefaults: Output, debugName: Output, contextSparseTypes: Array<Long> = arrayOf(), featureListDenseTypes: Array<Long> = arrayOf(), contextDenseShapes: Array<Shape> = arrayOf(), featureListSparseTypes: Array<Long> = arrayOf(), featureListDenseShapes: Array<Shape> = arrayOf(), name: String = "ParseSingleSequenceExample"): List<Output> {
      return gen_parsing_ops.parseSingleSequenceExample(serialized, featureListDenseMissingAssumedEmpty, contextSparseKeys, contextDenseKeys, featureListSparseKeys, featureListDenseKeys, contextDenseDefaults, debugName, contextSparseTypes, featureListDenseTypes, contextDenseShapes, featureListSparseTypes, featureListDenseShapes, name)
    }
    
    fun parseTensor(serialized: Output, outType: DataType<*>, name: String = "ParseTensor"): Output {
      return gen_parsing_ops.parseTensor(serialized, outType, name)
    }
    
    fun serializeTensor(tensor: Output, name: String = "SerializeTensor"): Output {
      return gen_parsing_ops.serializeTensor(tensor, name)
    }
    
    fun stringToNumber(stringTensor: Output, outType: DataType<*> = FLOAT, name: String = "StringToNumber"): Output {
      return gen_parsing_ops.stringToNumber(stringTensor, outType, name)
    }
  }
}