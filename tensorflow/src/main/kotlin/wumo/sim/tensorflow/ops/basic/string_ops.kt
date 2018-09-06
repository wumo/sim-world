package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_string_ops

object string_ops {
  interface API {
    fun asString(input: Output, precision: Long = -1L, scientific: Boolean = false, shortest: Boolean = false, width: Long = -1L, fill: String = "", name: String = "AsString"): Output {
      return gen_string_ops.asString(input, precision, scientific, shortest, width, fill, name)
    }
    
    fun decodeBase64(input: Output, name: String = "DecodeBase64"): Output {
      return gen_string_ops.decodeBase64(input, name)
    }
    
    fun encodeBase64(input: Output, pad: Boolean = false, name: String = "EncodeBase64"): Output {
      return gen_string_ops.encodeBase64(input, pad, name)
    }
    
    fun reduceJoin(inputs: Output, reductionIndices: Output, keepDims: Boolean = false, separator: String = "", name: String = "ReduceJoin"): Output {
      return gen_string_ops.reduceJoin(inputs, reductionIndices, keepDims, separator, name)
    }
    
    fun regexFullMatch(input: Output, pattern: Output, name: String = "RegexFullMatch"): Output {
      return gen_string_ops.regexFullMatch(input, pattern, name)
    }
    
    fun regexReplace(input: Output, pattern: Output, rewrite: Output, replaceGlobal: Boolean = true, name: String = "RegexReplace"): Output {
      return gen_string_ops.regexReplace(input, pattern, rewrite, replaceGlobal, name)
    }
    
    fun stringJoin(inputs: List<Output>, separator: String = "", name: String = "StringJoin"): Output {
      return gen_string_ops.stringJoin(inputs, separator, name)
    }
    
    fun stringSplit(input: Output, delimiter: Output, skipEmpty: Boolean = true, name: String = "StringSplit"): List<Output> {
      return gen_string_ops.stringSplit(input, delimiter, skipEmpty, name)
    }
    
    fun stringSplitV2(input: Output, sep: Output, maxsplit: Long = -1L, name: String = "StringSplitV2"): List<Output> {
      return gen_string_ops.stringSplitV2(input, sep, maxsplit, name)
    }
    
    fun stringStrip(input: Output, name: String = "StringStrip"): Output {
      return gen_string_ops.stringStrip(input, name)
    }
    
    fun stringToHashBucket(stringTensor: Output, numBuckets: Long, name: String = "StringToHashBucket"): Output {
      return gen_string_ops.stringToHashBucket(stringTensor, numBuckets, name)
    }
    
    fun stringToHashBucketFast(input: Output, numBuckets: Long, name: String = "StringToHashBucketFast"): Output {
      return gen_string_ops.stringToHashBucketFast(input, numBuckets, name)
    }
    
    fun stringToHashBucketStrong(input: Output, numBuckets: Long, key: Array<Long>, name: String = "StringToHashBucketStrong"): Output {
      return gen_string_ops.stringToHashBucketStrong(input, numBuckets, key, name)
    }
    
    fun substr(input: Output, pos: Output, len: Output, name: String = "Substr"): Output {
      return gen_string_ops.substr(input, pos, len, name)
    }
  }
}