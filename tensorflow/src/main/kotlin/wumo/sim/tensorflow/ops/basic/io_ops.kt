package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_io_ops
import wumo.sim.tensorflow.types.DataType

object io_ops {
  interface API {
    fun fixedLengthRecordReader(recordBytes: Long, headerBytes: Long = 0L, footerBytes: Long = 0L, hopBytes: Long = 0L, container: String = "", sharedName: String = "", name: String = "FixedLengthRecordReader"): Output {
      return gen_io_ops.fixedLengthRecordReader(recordBytes, headerBytes, footerBytes, hopBytes, container, sharedName, name)
    }
    
    fun fixedLengthRecordReaderV2(recordBytes: Long, headerBytes: Long = 0L, footerBytes: Long = 0L, hopBytes: Long = 0L, container: String = "", sharedName: String = "", encoding: String = "", name: String = "FixedLengthRecordReaderV2"): Output {
      return gen_io_ops.fixedLengthRecordReaderV2(recordBytes, headerBytes, footerBytes, hopBytes, container, sharedName, encoding, name)
    }
    
    fun identityReader(container: String = "", sharedName: String = "", name: String = "IdentityReader"): Output {
      return gen_io_ops.identityReader(container, sharedName, name)
    }
    
    fun identityReaderV2(container: String = "", sharedName: String = "", name: String = "IdentityReaderV2"): Output {
      return gen_io_ops.identityReaderV2(container, sharedName, name)
    }
    
    fun lMDBReader(container: String = "", sharedName: String = "", name: String = "LMDBReader"): Output {
      return gen_io_ops.lMDBReader(container, sharedName, name)
    }
    
    fun matchingFiles(pattern: Output, name: String = "MatchingFiles"): Output {
      return gen_io_ops.matchingFiles(pattern, name)
    }
    
    fun mergeV2Checkpoints(checkpointPrefixes: Output, destinationPrefix: Output, deleteOldDirs: Boolean = true, name: String = "MergeV2Checkpoints"): Op {
      return gen_io_ops.mergeV2Checkpoints(checkpointPrefixes, destinationPrefix, deleteOldDirs, name)
    }
    
    fun readFile(filename: Output, name: String = "ReadFile"): Output {
      return gen_io_ops.readFile(filename, name)
    }
    
    fun readerNumRecordsProduced(readerHandle: Output, name: String = "ReaderNumRecordsProduced"): Output {
      return gen_io_ops.readerNumRecordsProduced(readerHandle, name)
    }
    
    fun readerNumRecordsProducedV2(readerHandle: Output, name: String = "ReaderNumRecordsProducedV2"): Output {
      return gen_io_ops.readerNumRecordsProducedV2(readerHandle, name)
    }
    
    fun readerNumWorkUnitsCompleted(readerHandle: Output, name: String = "ReaderNumWorkUnitsCompleted"): Output {
      return gen_io_ops.readerNumWorkUnitsCompleted(readerHandle, name)
    }
    
    fun readerNumWorkUnitsCompletedV2(readerHandle: Output, name: String = "ReaderNumWorkUnitsCompletedV2"): Output {
      return gen_io_ops.readerNumWorkUnitsCompletedV2(readerHandle, name)
    }
    
    fun readerRead(readerHandle: Output, queueHandle: Output, name: String = "ReaderRead"): List<Output> {
      return gen_io_ops.readerRead(readerHandle, queueHandle, name)
    }
    
    fun readerReadUpTo(readerHandle: Output, queueHandle: Output, numRecords: Output, name: String = "ReaderReadUpTo"): List<Output> {
      return gen_io_ops.readerReadUpTo(readerHandle, queueHandle, numRecords, name)
    }
    
    fun readerReadUpToV2(readerHandle: Output, queueHandle: Output, numRecords: Output, name: String = "ReaderReadUpToV2"): List<Output> {
      return gen_io_ops.readerReadUpToV2(readerHandle, queueHandle, numRecords, name)
    }
    
    fun readerReadV2(readerHandle: Output, queueHandle: Output, name: String = "ReaderReadV2"): List<Output> {
      return gen_io_ops.readerReadV2(readerHandle, queueHandle, name)
    }
    
    fun readerReset(readerHandle: Output, name: String = "ReaderReset"): Op {
      return gen_io_ops.readerReset(readerHandle, name)
    }
    
    fun readerResetV2(readerHandle: Output, name: String = "ReaderResetV2"): Op {
      return gen_io_ops.readerResetV2(readerHandle, name)
    }
    
    fun readerRestoreState(readerHandle: Output, state: Output, name: String = "ReaderRestoreState"): Op {
      return gen_io_ops.readerRestoreState(readerHandle, state, name)
    }
    
    fun readerRestoreStateV2(readerHandle: Output, state: Output, name: String = "ReaderRestoreStateV2"): Op {
      return gen_io_ops.readerRestoreStateV2(readerHandle, state, name)
    }
    
    fun readerSerializeState(readerHandle: Output, name: String = "ReaderSerializeState"): Output {
      return gen_io_ops.readerSerializeState(readerHandle, name)
    }
    
    fun readerSerializeStateV2(readerHandle: Output, name: String = "ReaderSerializeStateV2"): Output {
      return gen_io_ops.readerSerializeStateV2(readerHandle, name)
    }
    
    fun restore(filePattern: Output, tensorName: Output, dt: DataType<*>, preferredShard: Long = -1L, name: String = "Restore"): Output {
      return gen_io_ops.restore(filePattern, tensorName, dt, preferredShard, name)
    }
    
    fun restoreSlice(filePattern: Output, tensorName: Output, shapeAndSlice: Output, dt: DataType<*>, preferredShard: Long = -1L, name: String = "RestoreSlice"): Output {
      return gen_io_ops.restoreSlice(filePattern, tensorName, shapeAndSlice, dt, preferredShard, name)
    }
    
    fun restoreV2(prefix: Output, tensorNames: Output, shapeAndSlices: Output, dtypes: Array<Long>, name: String = "RestoreV2"): List<Output> {
      return gen_io_ops.restoreV2(prefix, tensorNames, shapeAndSlices, dtypes, name)
    }
    
    fun save(filename: Output, tensorNames: Output, data: Output, name: String = "Save"): Op {
      return gen_io_ops.save(filename, tensorNames, data, name)
    }
    
    fun saveSlices(filename: Output, tensorNames: Output, shapesAndSlices: Output, data: Output, name: String = "SaveSlices"): Op {
      return gen_io_ops.saveSlices(filename, tensorNames, shapesAndSlices, data, name)
    }
    
    fun saveV2(prefix: Output, tensorNames: Output, shapeAndSlices: Output, tensors: Output, name: String = "SaveV2"): Op {
      return gen_io_ops.saveV2(prefix, tensorNames, shapeAndSlices, tensors, name)
    }
    
    fun shardedFilename(basename: Output, shard: Output, numShards: Output, name: String = "ShardedFilename"): Output {
      return gen_io_ops.shardedFilename(basename, shard, numShards, name)
    }
    
    fun shardedFilespec(basename: Output, numShards: Output, name: String = "ShardedFilespec"): Output {
      return gen_io_ops.shardedFilespec(basename, numShards, name)
    }
    
    fun tFRecordReader(container: String = "", sharedName: String = "", compressionType: String = "", name: String = "TFRecordReader"): Output {
      return gen_io_ops.tFRecordReader(container, sharedName, compressionType, name)
    }
    
    fun tFRecordReaderV2(container: String = "", sharedName: String = "", compressionType: String = "", name: String = "TFRecordReaderV2"): Output {
      return gen_io_ops.tFRecordReaderV2(container, sharedName, compressionType, name)
    }
    
    fun textLineReader(skipHeaderLines: Long = 0L, container: String = "", sharedName: String = "", name: String = "TextLineReader"): Output {
      return gen_io_ops.textLineReader(skipHeaderLines, container, sharedName, name)
    }
    
    fun textLineReaderV2(skipHeaderLines: Long = 0L, container: String = "", sharedName: String = "", name: String = "TextLineReaderV2"): Output {
      return gen_io_ops.textLineReaderV2(skipHeaderLines, container, sharedName, name)
    }
    
    fun wholeFileReader(container: String = "", sharedName: String = "", name: String = "WholeFileReader"): Output {
      return gen_io_ops.wholeFileReader(container, sharedName, name)
    }
    
    fun wholeFileReaderV2(container: String = "", sharedName: String = "", name: String = "WholeFileReaderV2"): Output {
      return gen_io_ops.wholeFileReaderV2(container, sharedName, name)
    }
    
    fun writeFile(filename: Output, contents: Output, name: String = "WriteFile"): Op {
      return gen_io_ops.writeFile(filename, contents, name)
    }
  }
}