/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.NameAttrList
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output
import wumo.sim.util.Shape

interface gen_dataset_ops {
  fun anonymousIterator(output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "AnonymousIterator") = run {
    buildOpTensor("AnonymousIterator", name) {
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun batchDataset(input_dataset: Output, batch_size: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "BatchDataset") = run {
    buildOpTensor("BatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun bytesProducedStatsDataset(input_dataset: Output, tag: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "BytesProducedStatsDataset") = run {
    buildOpTensor("BytesProducedStatsDataset", name) {
      addInput(input_dataset, false)
      addInput(tag, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun cacheDataset(input_dataset: Output, filename: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "CacheDataset") = run {
    buildOpTensor("CacheDataset", name) {
      addInput(input_dataset, false)
      addInput(filename, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun concatenateDataset(input_dataset: Output, another_dataset: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "ConcatenateDataset") = run {
    buildOpTensor("ConcatenateDataset", name) {
      addInput(input_dataset, false)
      addInput(another_dataset, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun datasetToSingleElement(dataset: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "DatasetToSingleElement") = run {
    buildOpTensors("DatasetToSingleElement", name) {
      addInput(dataset, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun denseToSparseBatchDataset(input_dataset: Output, batch_size: Output, row_shape: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "DenseToSparseBatchDataset") = run {
    buildOpTensor("DenseToSparseBatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(row_shape, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun deserializeIterator(resource_handle: Output, serialized: Output, name: String = "DeserializeIterator") = run {
    buildOp("DeserializeIterator", name) {
      addInput(resource_handle, false)
      addInput(serialized, false)
    }
  }
  
  fun enqueueInQueueDataset(queue: Output, components: Output, name: String = "EnqueueInQueueDataset") = run {
    buildOp("EnqueueInQueueDataset", name) {
      addInput(queue, false)
      addInput(components, false)
    }
  }
  
  fun featureStatsDataset(input_dataset: Output, tag: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "FeatureStatsDataset") = run {
    buildOpTensor("FeatureStatsDataset", name) {
      addInput(input_dataset, false)
      addInput(tag, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun filterDataset(input_dataset: Output, other_arguments: Output, predicate: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "FilterDataset") = run {
    buildOpTensor("FilterDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      attr("predicate", predicate)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun fixedLengthRecordDataset(filenames: Output, header_bytes: Output, record_bytes: Output, footer_bytes: Output, buffer_size: Output, name: String = "FixedLengthRecordDataset") = run {
    buildOpTensor("FixedLengthRecordDataset", name) {
      addInput(filenames, false)
      addInput(header_bytes, false)
      addInput(record_bytes, false)
      addInput(footer_bytes, false)
      addInput(buffer_size, false)
    }
  }
  
  fun flatMapDataset(input_dataset: Output, other_arguments: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "FlatMapDataset") = run {
    buildOpTensor("FlatMapDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun generatorDataset(init_func_other_args: Output, next_func_other_args: Output, finalize_func_other_args: Output, init_func: NameAttrList, next_func: NameAttrList, finalize_func: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "GeneratorDataset") = run {
    buildOpTensor("GeneratorDataset", name) {
      addInput(init_func_other_args, false)
      addInput(next_func_other_args, false)
      addInput(finalize_func_other_args, false)
      attr("init_func", init_func)
      attr("next_func", next_func)
      attr("finalize_func", finalize_func)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun groupByWindowDataset(input_dataset: Output, key_func_other_arguments: Output, reduce_func_other_arguments: Output, window_size_func_other_arguments: Output, key_func: NameAttrList, reduce_func: NameAttrList, window_size_func: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "GroupByWindowDataset") = run {
    buildOpTensor("GroupByWindowDataset", name) {
      addInput(input_dataset, false)
      addInput(key_func_other_arguments, false)
      addInput(reduce_func_other_arguments, false)
      addInput(window_size_func_other_arguments, false)
      attr("key_func", key_func)
      attr("reduce_func", reduce_func)
      attr("window_size_func", window_size_func)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun interleaveDataset(input_dataset: Output, other_arguments: Output, cycle_length: Output, block_length: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "InterleaveDataset") = run {
    buildOpTensor("InterleaveDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      addInput(cycle_length, false)
      addInput(block_length, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iterator(shared_name: String, container: String, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "Iterator") = run {
    buildOpTensor("Iterator", name) {
      attr("shared_name", shared_name)
      attr("container", container)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorFromStringHandle(string_handle: Output, output_types: Array<Long> = arrayOf(), output_shapes: Array<Shape> = arrayOf(), name: String = "IteratorFromStringHandle") = run {
    buildOpTensor("IteratorFromStringHandle", name) {
      addInput(string_handle, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorGetNext(iterator: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "IteratorGetNext") = run {
    buildOpTensors("IteratorGetNext", name) {
      addInput(iterator, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorGetNextSync(iterator: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "IteratorGetNextSync") = run {
    buildOpTensors("IteratorGetNextSync", name) {
      addInput(iterator, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorToStringHandle(resource_handle: Output, name: String = "IteratorToStringHandle") = run {
    buildOpTensor("IteratorToStringHandle", name) {
      addInput(resource_handle, false)
    }
  }
  
  fun latencyStatsDataset(input_dataset: Output, tag: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "LatencyStatsDataset") = run {
    buildOpTensor("LatencyStatsDataset", name) {
      addInput(input_dataset, false)
      addInput(tag, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun makeIterator(dataset: Output, iterator: Output, name: String = "MakeIterator") = run {
    buildOp("MakeIterator", name) {
      addInput(dataset, false)
      addInput(iterator, false)
    }
  }
  
  fun mapDataset(input_dataset: Output, other_arguments: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "MapDataset") = run {
    buildOpTensor("MapDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun oneShotIterator(dataset_factory: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, container: String = "", shared_name: String = "", name: String = "OneShotIterator") = run {
    buildOpTensor("OneShotIterator", name) {
      attr("dataset_factory", dataset_factory)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun paddedBatchDataset(input_dataset: Output, batch_size: Output, padded_shapes: List<Output>, padding_values: Output, output_shapes: Array<Shape>, name: String = "PaddedBatchDataset") = run {
    buildOpTensor("PaddedBatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(padded_shapes, false)
      addInput(padding_values, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun parallelInterleaveDataset(input_dataset: Output, other_arguments: Output, cycle_length: Output, block_length: Output, sloppy: Output, buffer_output_elements: Output, prefetch_input_elements: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "ParallelInterleaveDataset") = run {
    buildOpTensor("ParallelInterleaveDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      addInput(cycle_length, false)
      addInput(block_length, false)
      addInput(sloppy, false)
      addInput(buffer_output_elements, false)
      addInput(prefetch_input_elements, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun parallelMapDataset(input_dataset: Output, other_arguments: Output, num_parallel_calls: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "ParallelMapDataset") = run {
    buildOpTensor("ParallelMapDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      addInput(num_parallel_calls, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun prefetchDataset(input_dataset: Output, buffer_size: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "PrefetchDataset") = run {
    buildOpTensor("PrefetchDataset", name) {
      addInput(input_dataset, false)
      addInput(buffer_size, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun prependFromQueueAndPaddedBatchDataset(input_dataset: Output, batch_size: Output, padded_shapes: List<Output>, padding_values: Output, output_shapes: Array<Shape>, name: String = "PrependFromQueueAndPaddedBatchDataset") = run {
    buildOpTensor("PrependFromQueueAndPaddedBatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(padded_shapes, false)
      addInput(padding_values, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun randomDataset(seed: Output, seed2: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "RandomDataset") = run {
    buildOpTensor("RandomDataset", name) {
      addInput(seed, false)
      addInput(seed2, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun rangeDataset(start: Output, stop: Output, step: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "RangeDataset") = run {
    buildOpTensor("RangeDataset", name) {
      addInput(start, false)
      addInput(stop, false)
      addInput(step, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun repeatDataset(input_dataset: Output, count: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "RepeatDataset") = run {
    buildOpTensor("RepeatDataset", name) {
      addInput(input_dataset, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun scanDataset(input_dataset: Output, initial_state: Output, other_arguments: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "ScanDataset") = run {
    buildOpTensor("ScanDataset", name) {
      addInput(input_dataset, false)
      addInput(initial_state, false)
      addInput(other_arguments, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun serializeIterator(resource_handle: Output, name: String = "SerializeIterator") = run {
    buildOpTensor("SerializeIterator", name) {
      addInput(resource_handle, false)
    }
  }
  
  fun setStatsAggregatorDataset(input_dataset: Output, stats_aggregator: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "SetStatsAggregatorDataset") = run {
    buildOpTensor("SetStatsAggregatorDataset", name) {
      addInput(input_dataset, false)
      addInput(stats_aggregator, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun shuffleAndRepeatDataset(input_dataset: Output, buffer_size: Output, seed: Output, seed2: Output, count: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "ShuffleAndRepeatDataset") = run {
    buildOpTensor("ShuffleAndRepeatDataset", name) {
      addInput(input_dataset, false)
      addInput(buffer_size, false)
      addInput(seed, false)
      addInput(seed2, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun shuffleDataset(input_dataset: Output, buffer_size: Output, seed: Output, seed2: Output, output_types: Array<Long>, output_shapes: Array<Shape>, reshuffle_each_iteration: Boolean = true, name: String = "ShuffleDataset") = run {
    buildOpTensor("ShuffleDataset", name) {
      addInput(input_dataset, false)
      addInput(buffer_size, false)
      addInput(seed, false)
      addInput(seed2, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
      attr("reshuffle_each_iteration", reshuffle_each_iteration)
    }
  }
  
  fun skipDataset(input_dataset: Output, count: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "SkipDataset") = run {
    buildOpTensor("SkipDataset", name) {
      addInput(input_dataset, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun slideDataset(input_dataset: Output, window_size: Output, window_shift: Output, window_stride: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "SlideDataset") = run {
    buildOpTensor("SlideDataset", name) {
      addInput(input_dataset, false)
      addInput(window_size, false)
      addInput(window_shift, false)
      addInput(window_stride, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun sparseTensorSliceDataset(indices: Output, values: Output, dense_shape: Output, name: String = "SparseTensorSliceDataset") = run {
    buildOpTensor("SparseTensorSliceDataset", name) {
      addInput(indices, false)
      addInput(values, false)
      addInput(dense_shape, false)
    }
  }
  
  fun sqlDataset(driver_name: Output, data_source_name: Output, query: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "SqlDataset") = run {
    buildOpTensor("SqlDataset", name) {
      addInput(driver_name, false)
      addInput(data_source_name, false)
      addInput(query, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun statsAggregatorHandle(container: String = "", shared_name: String = "", name: String = "StatsAggregatorHandle") = run {
    buildOpTensor("StatsAggregatorHandle", name) {
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun statsAggregatorSummary(iterator: Output, name: String = "StatsAggregatorSummary") = run {
    buildOpTensor("StatsAggregatorSummary", name) {
      addInput(iterator, false)
    }
  }
  
  fun tFRecordDataset(filenames: Output, compression_type: Output, buffer_size: Output, name: String = "TFRecordDataset") = run {
    buildOpTensor("TFRecordDataset", name) {
      addInput(filenames, false)
      addInput(compression_type, false)
      addInput(buffer_size, false)
    }
  }
  
  fun takeDataset(input_dataset: Output, count: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "TakeDataset") = run {
    buildOpTensor("TakeDataset", name) {
      addInput(input_dataset, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun tensorDataset(components: Output, output_shapes: Array<Shape>, name: String = "TensorDataset") = run {
    buildOpTensor("TensorDataset", name) {
      addInput(components, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun tensorSliceDataset(components: Output, output_shapes: Array<Shape>, name: String = "TensorSliceDataset") = run {
    buildOpTensor("TensorSliceDataset", name) {
      addInput(components, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun textLineDataset(filenames: Output, compression_type: Output, buffer_size: Output, name: String = "TextLineDataset") = run {
    buildOpTensor("TextLineDataset", name) {
      addInput(filenames, false)
      addInput(compression_type, false)
      addInput(buffer_size, false)
    }
  }
  
  fun unbatchDataset(input_dataset: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "UnbatchDataset") = run {
    buildOpTensor("UnbatchDataset", name) {
      addInput(input_dataset, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun zipDataset(input_datasets: List<Output>, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "ZipDataset") = run {
    buildOpTensor("ZipDataset", name) {
      addInput(input_datasets, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun batchDatasetV2(input_dataset: Output, batch_size: Output, drop_remainder: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "BatchDatasetV2") = run {
    buildOpTensor("BatchDatasetV2", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(drop_remainder, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun datasetToGraph(input_dataset: Output, name: String = "DatasetToGraph") = run {
    buildOpTensor("DatasetToGraph", name) {
      addInput(input_dataset, false)
    }
  }
  
  fun datasetToTFRecord(input_dataset: Output, filename: Output, compression_type: Output, name: String = "DatasetToTFRecord") = run {
    buildOp("DatasetToTFRecord", name) {
      addInput(input_dataset, false)
      addInput(filename, false)
      addInput(compression_type, false)
    }
  }
  
  fun groupByReducerDataset(input_dataset: Output, key_func_other_arguments: Output, init_func_other_arguments: Output, reduce_func_other_arguments: Output, finalize_func_other_arguments: Output, key_func: NameAttrList, init_func: NameAttrList, reduce_func: NameAttrList, finalize_func: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "GroupByReducerDataset") = run {
    buildOpTensor("GroupByReducerDataset", name) {
      addInput(input_dataset, false)
      addInput(key_func_other_arguments, false)
      addInput(init_func_other_arguments, false)
      addInput(reduce_func_other_arguments, false)
      addInput(finalize_func_other_arguments, false)
      attr("key_func", key_func)
      attr("init_func", init_func)
      attr("reduce_func", reduce_func)
      attr("finalize_func", finalize_func)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorFromStringHandleV2(string_handle: Output, output_types: Array<Long> = arrayOf(), output_shapes: Array<Shape> = arrayOf(), name: String = "IteratorFromStringHandleV2") = run {
    buildOpTensor("IteratorFromStringHandleV2", name) {
      addInput(string_handle, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorV2(shared_name: String, container: String, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "IteratorV2") = run {
    buildOpTensor("IteratorV2", name) {
      attr("shared_name", shared_name)
      attr("container", container)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun mapAndBatchDataset(input_dataset: Output, other_arguments: Output, batch_size: Output, num_parallel_batches: Output, drop_remainder: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "MapAndBatchDataset") = run {
    buildOpTensor("MapAndBatchDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      addInput(batch_size, false)
      addInput(num_parallel_batches, false)
      addInput(drop_remainder, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun mapAndBatchDatasetV2(input_dataset: Output, other_arguments: Output, batch_size: Output, num_parallel_calls: Output, drop_remainder: Output, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "MapAndBatchDatasetV2") = run {
    buildOpTensor("MapAndBatchDatasetV2", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      addInput(batch_size, false)
      addInput(num_parallel_calls, false)
      addInput(drop_remainder, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun optimizeDataset(input_dataset: Output, optimizations: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "OptimizeDataset") = run {
    buildOpTensor("OptimizeDataset", name) {
      addInput(input_dataset, false)
      addInput(optimizations, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun paddedBatchDatasetV2(input_dataset: Output, batch_size: Output, padded_shapes: List<Output>, padding_values: Output, drop_remainder: Output, output_shapes: Array<Shape>, name: String = "PaddedBatchDatasetV2") = run {
    buildOpTensor("PaddedBatchDatasetV2", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(padded_shapes, false)
      addInput(padding_values, false)
      addInput(drop_remainder, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun sinkDataset(input_dataset: Output, name: String = "SinkDataset") = run {
    buildOpTensor("SinkDataset", name) {
      addInput(input_dataset, false)
    }
  }
  
  fun windowDataset(input_dataset: Output, window_size: Output, output_types: Array<Long>, output_shapes: Array<Shape>, name: String = "WindowDataset") = run {
    buildOpTensor("WindowDataset", name) {
      addInput(input_dataset, false)
      addInput(window_size, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
}