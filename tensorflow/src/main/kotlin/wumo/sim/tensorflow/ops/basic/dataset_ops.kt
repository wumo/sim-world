package wumo.sim.tensorflow.ops.basic

import org.bytedeco.javacpp.tensorflow.NameAttrList
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_dataset_ops
import wumo.sim.util.Shape

object dataset_ops {
  interface API {
    fun anonymousIterator(outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "AnonymousIterator"): Output {
      return gen_dataset_ops.anonymousIterator(outputTypes, outputShapes, name)
    }
    
    fun batchDataset(inputDataset: Output, batchSize: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "BatchDataset"): Output {
      return gen_dataset_ops.batchDataset(inputDataset, batchSize, outputTypes, outputShapes, name)
    }
    
    fun batchDatasetV2(inputDataset: Output, batchSize: Output, dropRemainder: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "BatchDatasetV2"): Output {
      return gen_dataset_ops.batchDatasetV2(inputDataset, batchSize, dropRemainder, outputTypes, outputShapes, name)
    }
    
    fun bytesProducedStatsDataset(inputDataset: Output, tag: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "BytesProducedStatsDataset"): Output {
      return gen_dataset_ops.bytesProducedStatsDataset(inputDataset, tag, outputTypes, outputShapes, name)
    }
    
    fun cacheDataset(inputDataset: Output, filename: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "CacheDataset"): Output {
      return gen_dataset_ops.cacheDataset(inputDataset, filename, outputTypes, outputShapes, name)
    }
    
    fun concatenateDataset(inputDataset: Output, anotherDataset: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "ConcatenateDataset"): Output {
      return gen_dataset_ops.concatenateDataset(inputDataset, anotherDataset, outputTypes, outputShapes, name)
    }
    
    fun datasetToGraph(inputDataset: Output, name: String = "DatasetToGraph"): Output {
      return gen_dataset_ops.datasetToGraph(inputDataset, name)
    }
    
    fun datasetToSingleElement(dataset: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "DatasetToSingleElement"): List<Output> {
      return gen_dataset_ops.datasetToSingleElement(dataset, outputTypes, outputShapes, name)
    }
    
    fun datasetToTFRecord(inputDataset: Output, filename: Output, compressionType: Output, name: String = "DatasetToTFRecord"): Op {
      return gen_dataset_ops.datasetToTFRecord(inputDataset, filename, compressionType, name)
    }
    
    fun denseToSparseBatchDataset(inputDataset: Output, batchSize: Output, rowShape: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "DenseToSparseBatchDataset"): Output {
      return gen_dataset_ops.denseToSparseBatchDataset(inputDataset, batchSize, rowShape, outputTypes, outputShapes, name)
    }
    
    fun deserializeIterator(resourceHandle: Output, serialized: Output, name: String = "DeserializeIterator"): Op {
      return gen_dataset_ops.deserializeIterator(resourceHandle, serialized, name)
    }
    
    fun enqueueInQueueDataset(queue: Output, components: Output, name: String = "EnqueueInQueueDataset"): Op {
      return gen_dataset_ops.enqueueInQueueDataset(queue, components, name)
    }
    
    fun featureStatsDataset(inputDataset: Output, tag: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "FeatureStatsDataset"): Output {
      return gen_dataset_ops.featureStatsDataset(inputDataset, tag, outputTypes, outputShapes, name)
    }
    
    fun filterDataset(inputDataset: Output, otherArguments: Output, predicate: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "FilterDataset"): Output {
      return gen_dataset_ops.filterDataset(inputDataset, otherArguments, predicate, outputTypes, outputShapes, name)
    }
    
    fun fixedLengthRecordDataset(filenames: Output, headerBytes: Output, recordBytes: Output, footerBytes: Output, bufferSize: Output, name: String = "FixedLengthRecordDataset"): Output {
      return gen_dataset_ops.fixedLengthRecordDataset(filenames, headerBytes, recordBytes, footerBytes, bufferSize, name)
    }
    
    fun flatMapDataset(inputDataset: Output, otherArguments: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "FlatMapDataset"): Output {
      return gen_dataset_ops.flatMapDataset(inputDataset, otherArguments, f, outputTypes, outputShapes, name)
    }
    
    fun generatorDataset(initFuncOtherArgs: Output, nextFuncOtherArgs: Output, finalizeFuncOtherArgs: Output, initFunc: NameAttrList, nextFunc: NameAttrList, finalizeFunc: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "GeneratorDataset"): Output {
      return gen_dataset_ops.generatorDataset(initFuncOtherArgs, nextFuncOtherArgs, finalizeFuncOtherArgs, initFunc, nextFunc, finalizeFunc, outputTypes, outputShapes, name)
    }
    
    fun groupByReducerDataset(inputDataset: Output, keyFuncOtherArguments: Output, initFuncOtherArguments: Output, reduceFuncOtherArguments: Output, finalizeFuncOtherArguments: Output, keyFunc: NameAttrList, initFunc: NameAttrList, reduceFunc: NameAttrList, finalizeFunc: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "GroupByReducerDataset"): Output {
      return gen_dataset_ops.groupByReducerDataset(inputDataset, keyFuncOtherArguments, initFuncOtherArguments, reduceFuncOtherArguments, finalizeFuncOtherArguments, keyFunc, initFunc, reduceFunc, finalizeFunc, outputTypes, outputShapes, name)
    }
    
    fun groupByWindowDataset(inputDataset: Output, keyFuncOtherArguments: Output, reduceFuncOtherArguments: Output, windowSizeFuncOtherArguments: Output, keyFunc: NameAttrList, reduceFunc: NameAttrList, windowSizeFunc: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "GroupByWindowDataset"): Output {
      return gen_dataset_ops.groupByWindowDataset(inputDataset, keyFuncOtherArguments, reduceFuncOtherArguments, windowSizeFuncOtherArguments, keyFunc, reduceFunc, windowSizeFunc, outputTypes, outputShapes, name)
    }
    
    fun interleaveDataset(inputDataset: Output, otherArguments: Output, cycleLength: Output, blockLength: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "InterleaveDataset"): Output {
      return gen_dataset_ops.interleaveDataset(inputDataset, otherArguments, cycleLength, blockLength, f, outputTypes, outputShapes, name)
    }
    
    fun iterator(sharedName: String, container: String, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "Iterator"): Output {
      return gen_dataset_ops.iterator(sharedName, container, outputTypes, outputShapes, name)
    }
    
    fun iteratorFromStringHandle(stringHandle: Output, outputTypes: Array<Long> = arrayOf(), outputShapes: Array<Shape> = arrayOf(), name: String = "IteratorFromStringHandle"): Output {
      return gen_dataset_ops.iteratorFromStringHandle(stringHandle, outputTypes, outputShapes, name)
    }
    
    fun iteratorFromStringHandleV2(stringHandle: Output, outputTypes: Array<Long> = arrayOf(), outputShapes: Array<Shape> = arrayOf(), name: String = "IteratorFromStringHandleV2"): Output {
      return gen_dataset_ops.iteratorFromStringHandleV2(stringHandle, outputTypes, outputShapes, name)
    }
    
    fun iteratorGetNext(iterator: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "IteratorGetNext"): List<Output> {
      return gen_dataset_ops.iteratorGetNext(iterator, outputTypes, outputShapes, name)
    }
    
    fun iteratorGetNextSync(iterator: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "IteratorGetNextSync"): List<Output> {
      return gen_dataset_ops.iteratorGetNextSync(iterator, outputTypes, outputShapes, name)
    }
    
    fun iteratorToStringHandle(resourceHandle: Output, name: String = "IteratorToStringHandle"): Output {
      return gen_dataset_ops.iteratorToStringHandle(resourceHandle, name)
    }
    
    fun iteratorV2(sharedName: String, container: String, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "IteratorV2"): Output {
      return gen_dataset_ops.iteratorV2(sharedName, container, outputTypes, outputShapes, name)
    }
    
    fun latencyStatsDataset(inputDataset: Output, tag: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "LatencyStatsDataset"): Output {
      return gen_dataset_ops.latencyStatsDataset(inputDataset, tag, outputTypes, outputShapes, name)
    }
    
    fun makeIterator(dataset: Output, iterator: Output, name: String = "MakeIterator"): Op {
      return gen_dataset_ops.makeIterator(dataset, iterator, name)
    }
    
    fun mapAndBatchDataset(inputDataset: Output, otherArguments: Output, batchSize: Output, numParallelBatches: Output, dropRemainder: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "MapAndBatchDataset"): Output {
      return gen_dataset_ops.mapAndBatchDataset(inputDataset, otherArguments, batchSize, numParallelBatches, dropRemainder, f, outputTypes, outputShapes, name)
    }
    
    fun mapAndBatchDatasetV2(inputDataset: Output, otherArguments: Output, batchSize: Output, numParallelCalls: Output, dropRemainder: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "MapAndBatchDatasetV2"): Output {
      return gen_dataset_ops.mapAndBatchDatasetV2(inputDataset, otherArguments, batchSize, numParallelCalls, dropRemainder, f, outputTypes, outputShapes, name)
    }
    
    fun mapDataset(inputDataset: Output, otherArguments: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "MapDataset"): Output {
      return gen_dataset_ops.mapDataset(inputDataset, otherArguments, f, outputTypes, outputShapes, name)
    }
    
    fun oneShotIterator(datasetFactory: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, container: String = "", sharedName: String = "", name: String = "OneShotIterator"): Output {
      return gen_dataset_ops.oneShotIterator(datasetFactory, outputTypes, outputShapes, container, sharedName, name)
    }
    
    fun optimizeDataset(inputDataset: Output, optimizations: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "OptimizeDataset"): Output {
      return gen_dataset_ops.optimizeDataset(inputDataset, optimizations, outputTypes, outputShapes, name)
    }
    
    fun paddedBatchDataset(inputDataset: Output, batchSize: Output, paddedShapes: List<Output>, paddingValues: Output, outputShapes: Array<Shape>, name: String = "PaddedBatchDataset"): Output {
      return gen_dataset_ops.paddedBatchDataset(inputDataset, batchSize, paddedShapes, paddingValues, outputShapes, name)
    }
    
    fun paddedBatchDatasetV2(inputDataset: Output, batchSize: Output, paddedShapes: List<Output>, paddingValues: Output, dropRemainder: Output, outputShapes: Array<Shape>, name: String = "PaddedBatchDatasetV2"): Output {
      return gen_dataset_ops.paddedBatchDatasetV2(inputDataset, batchSize, paddedShapes, paddingValues, dropRemainder, outputShapes, name)
    }
    
    fun parallelInterleaveDataset(inputDataset: Output, otherArguments: Output, cycleLength: Output, blockLength: Output, sloppy: Output, bufferOutputElements: Output, prefetchInputElements: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "ParallelInterleaveDataset"): Output {
      return gen_dataset_ops.parallelInterleaveDataset(inputDataset, otherArguments, cycleLength, blockLength, sloppy, bufferOutputElements, prefetchInputElements, f, outputTypes, outputShapes, name)
    }
    
    fun parallelMapDataset(inputDataset: Output, otherArguments: Output, numParallelCalls: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "ParallelMapDataset"): Output {
      return gen_dataset_ops.parallelMapDataset(inputDataset, otherArguments, numParallelCalls, f, outputTypes, outputShapes, name)
    }
    
    fun prefetchDataset(inputDataset: Output, bufferSize: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "PrefetchDataset"): Output {
      return gen_dataset_ops.prefetchDataset(inputDataset, bufferSize, outputTypes, outputShapes, name)
    }
    
    fun prependFromQueueAndPaddedBatchDataset(inputDataset: Output, batchSize: Output, paddedShapes: List<Output>, paddingValues: Output, outputShapes: Array<Shape>, name: String = "PrependFromQueueAndPaddedBatchDataset"): Output {
      return gen_dataset_ops.prependFromQueueAndPaddedBatchDataset(inputDataset, batchSize, paddedShapes, paddingValues, outputShapes, name)
    }
    
    fun randomDataset(seed: Output, seed2: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "RandomDataset"): Output {
      return gen_dataset_ops.randomDataset(seed, seed2, outputTypes, outputShapes, name)
    }
    
    fun rangeDataset(start: Output, stop: Output, step: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "RangeDataset"): Output {
      return gen_dataset_ops.rangeDataset(start, stop, step, outputTypes, outputShapes, name)
    }
    
    fun repeatDataset(inputDataset: Output, count: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "RepeatDataset"): Output {
      return gen_dataset_ops.repeatDataset(inputDataset, count, outputTypes, outputShapes, name)
    }
    
    fun scanDataset(inputDataset: Output, initialState: Output, otherArguments: Output, f: NameAttrList, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "ScanDataset"): Output {
      return gen_dataset_ops.scanDataset(inputDataset, initialState, otherArguments, f, outputTypes, outputShapes, name)
    }
    
    fun serializeIterator(resourceHandle: Output, name: String = "SerializeIterator"): Output {
      return gen_dataset_ops.serializeIterator(resourceHandle, name)
    }
    
    fun setStatsAggregatorDataset(inputDataset: Output, statsAggregator: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "SetStatsAggregatorDataset"): Output {
      return gen_dataset_ops.setStatsAggregatorDataset(inputDataset, statsAggregator, outputTypes, outputShapes, name)
    }
    
    fun shuffleAndRepeatDataset(inputDataset: Output, bufferSize: Output, seed: Output, seed2: Output, count: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "ShuffleAndRepeatDataset"): Output {
      return gen_dataset_ops.shuffleAndRepeatDataset(inputDataset, bufferSize, seed, seed2, count, outputTypes, outputShapes, name)
    }
    
    fun shuffleDataset(inputDataset: Output, bufferSize: Output, seed: Output, seed2: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, reshuffleEachIteration: Boolean = true, name: String = "ShuffleDataset"): Output {
      return gen_dataset_ops.shuffleDataset(inputDataset, bufferSize, seed, seed2, outputTypes, outputShapes, reshuffleEachIteration, name)
    }
    
    fun sinkDataset(inputDataset: Output, name: String = "SinkDataset"): Output {
      return gen_dataset_ops.sinkDataset(inputDataset, name)
    }
    
    fun skipDataset(inputDataset: Output, count: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "SkipDataset"): Output {
      return gen_dataset_ops.skipDataset(inputDataset, count, outputTypes, outputShapes, name)
    }
    
    fun slideDataset(inputDataset: Output, windowSize: Output, stride: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "SlideDataset"): Output {
      return gen_dataset_ops.slideDataset(inputDataset, windowSize, stride, outputTypes, outputShapes, name)
    }
    
    fun sparseTensorSliceDataset(indices: Output, values: Output, denseShape: Output, name: String = "SparseTensorSliceDataset"): Output {
      return gen_dataset_ops.sparseTensorSliceDataset(indices, values, denseShape, name)
    }
    
    fun sqlDataset(driverName: Output, dataSourceName: Output, query: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "SqlDataset"): Output {
      return gen_dataset_ops.sqlDataset(driverName, dataSourceName, query, outputTypes, outputShapes, name)
    }
    
    fun statsAggregatorHandle(container: String = "", sharedName: String = "", name: String = "StatsAggregatorHandle"): Output {
      return gen_dataset_ops.statsAggregatorHandle(container, sharedName, name)
    }
    
    fun statsAggregatorSummary(iterator: Output, name: String = "StatsAggregatorSummary"): Output {
      return gen_dataset_ops.statsAggregatorSummary(iterator, name)
    }
    
    fun tFRecordDataset(filenames: Output, compressionType: Output, bufferSize: Output, name: String = "TFRecordDataset"): Output {
      return gen_dataset_ops.tFRecordDataset(filenames, compressionType, bufferSize, name)
    }
    
    fun takeDataset(inputDataset: Output, count: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "TakeDataset"): Output {
      return gen_dataset_ops.takeDataset(inputDataset, count, outputTypes, outputShapes, name)
    }
    
    fun tensorDataset(components: Output, outputShapes: Array<Shape>, name: String = "TensorDataset"): Output {
      return gen_dataset_ops.tensorDataset(components, outputShapes, name)
    }
    
    fun tensorSliceDataset(components: Output, outputShapes: Array<Shape>, name: String = "TensorSliceDataset"): Output {
      return gen_dataset_ops.tensorSliceDataset(components, outputShapes, name)
    }
    
    fun textLineDataset(filenames: Output, compressionType: Output, bufferSize: Output, name: String = "TextLineDataset"): Output {
      return gen_dataset_ops.textLineDataset(filenames, compressionType, bufferSize, name)
    }
    
    fun unbatchDataset(inputDataset: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "UnbatchDataset"): Output {
      return gen_dataset_ops.unbatchDataset(inputDataset, outputTypes, outputShapes, name)
    }
    
    fun windowDataset(inputDataset: Output, windowSize: Output, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "WindowDataset"): Output {
      return gen_dataset_ops.windowDataset(inputDataset, windowSize, outputTypes, outputShapes, name)
    }
    
    fun zipDataset(inputDatasets: List<Output>, outputTypes: Array<Long>, outputShapes: Array<Shape>, name: String = "ZipDataset"): Output {
      return gen_dataset_ops.zipDataset(inputDatasets, outputTypes, outputShapes, name)
    }
  }
}