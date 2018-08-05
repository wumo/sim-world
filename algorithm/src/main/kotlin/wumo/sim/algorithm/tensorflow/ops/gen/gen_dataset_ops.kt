/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.algorithm.tensorflow.ops.gen

import org.bytedeco.javacpp.tensorflow.NameAttrList
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.util.Dimension

object gen_dataset_ops {
  fun anonymousIterator(output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "AnonymousIterator") = run {
    tf.buildOpTensor("AnonymousIterator", name) {
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun batchDataset(input_dataset: Tensor, batch_size: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "BatchDataset") = run {
    tf.buildOpTensor("BatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun bytesProducedStatsDataset(input_dataset: Tensor, tag: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "BytesProducedStatsDataset") = run {
    tf.buildOpTensor("BytesProducedStatsDataset", name) {
      addInput(input_dataset, false)
      addInput(tag, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun cacheDataset(input_dataset: Tensor, filename: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "CacheDataset") = run {
    tf.buildOpTensor("CacheDataset", name) {
      addInput(input_dataset, false)
      addInput(filename, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun concatenateDataset(input_dataset: Tensor, another_dataset: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "ConcatenateDataset") = run {
    tf.buildOpTensor("ConcatenateDataset", name) {
      addInput(input_dataset, false)
      addInput(another_dataset, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun datasetToSingleElement(dataset: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "DatasetToSingleElement") = run {
    tf.buildOpTensors("DatasetToSingleElement", name) {
      addInput(dataset, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun denseToSparseBatchDataset(input_dataset: Tensor, batch_size: Tensor, row_shape: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "DenseToSparseBatchDataset") = run {
    tf.buildOpTensor("DenseToSparseBatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(row_shape, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun deserializeIterator(resource_handle: Tensor, serialized: Tensor, name: String = "DeserializeIterator") = run {
    tf.buildOp("DeserializeIterator", name) {
      addInput(resource_handle, false)
      addInput(serialized, false)
    }
  }
  
  fun enqueueInQueueDataset(queue: Tensor, components: Tensor, name: String = "EnqueueInQueueDataset") = run {
    tf.buildOp("EnqueueInQueueDataset", name) {
      addInput(queue, false)
      addInput(components, false)
    }
  }
  
  fun featureStatsDataset(input_dataset: Tensor, tag: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "FeatureStatsDataset") = run {
    tf.buildOpTensor("FeatureStatsDataset", name) {
      addInput(input_dataset, false)
      addInput(tag, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun filterDataset(input_dataset: Tensor, other_arguments: Tensor, predicate: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "FilterDataset") = run {
    tf.buildOpTensor("FilterDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      attr("predicate", predicate)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun fixedLengthRecordDataset(filenames: Tensor, header_bytes: Tensor, record_bytes: Tensor, footer_bytes: Tensor, buffer_size: Tensor, name: String = "FixedLengthRecordDataset") = run {
    tf.buildOpTensor("FixedLengthRecordDataset", name) {
      addInput(filenames, false)
      addInput(header_bytes, false)
      addInput(record_bytes, false)
      addInput(footer_bytes, false)
      addInput(buffer_size, false)
    }
  }
  
  fun flatMapDataset(input_dataset: Tensor, other_arguments: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "FlatMapDataset") = run {
    tf.buildOpTensor("FlatMapDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun generatorDataset(init_func_other_args: Tensor, next_func_other_args: Tensor, finalize_func_other_args: Tensor, init_func: NameAttrList, next_func: NameAttrList, finalize_func: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "GeneratorDataset") = run {
    tf.buildOpTensor("GeneratorDataset", name) {
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
  
  fun groupByWindowDataset(input_dataset: Tensor, key_func_other_arguments: Tensor, reduce_func_other_arguments: Tensor, window_size_func_other_arguments: Tensor, key_func: NameAttrList, reduce_func: NameAttrList, window_size_func: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "GroupByWindowDataset") = run {
    tf.buildOpTensor("GroupByWindowDataset", name) {
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
  
  fun interleaveDataset(input_dataset: Tensor, other_arguments: Tensor, cycle_length: Tensor, block_length: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "InterleaveDataset") = run {
    tf.buildOpTensor("InterleaveDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      addInput(cycle_length, false)
      addInput(block_length, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iterator(shared_name: String, container: String, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "Iterator") = run {
    tf.buildOpTensor("Iterator", name) {
      attr("shared_name", shared_name)
      attr("container", container)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorFromStringHandle(string_handle: Tensor, output_types: Array<Long> = arrayOf(), output_shapes: Array<Dimension> = arrayOf(), name: String = "IteratorFromStringHandle") = run {
    tf.buildOpTensor("IteratorFromStringHandle", name) {
      addInput(string_handle, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorGetNext(iterator: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "IteratorGetNext") = run {
    tf.buildOpTensors("IteratorGetNext", name) {
      addInput(iterator, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorGetNextSync(iterator: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "IteratorGetNextSync") = run {
    tf.buildOpTensors("IteratorGetNextSync", name) {
      addInput(iterator, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorToStringHandle(resource_handle: Tensor, name: String = "IteratorToStringHandle") = run {
    tf.buildOpTensor("IteratorToStringHandle", name) {
      addInput(resource_handle, false)
    }
  }
  
  fun latencyStatsDataset(input_dataset: Tensor, tag: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "LatencyStatsDataset") = run {
    tf.buildOpTensor("LatencyStatsDataset", name) {
      addInput(input_dataset, false)
      addInput(tag, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun makeIterator(dataset: Tensor, iterator: Tensor, name: String = "MakeIterator") = run {
    tf.buildOp("MakeIterator", name) {
      addInput(dataset, false)
      addInput(iterator, false)
    }
  }
  
  fun mapDataset(input_dataset: Tensor, other_arguments: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "MapDataset") = run {
    tf.buildOpTensor("MapDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun oneShotIterator(dataset_factory: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, container: String = "", shared_name: String = "", name: String = "OneShotIterator") = run {
    tf.buildOpTensor("OneShotIterator", name) {
      attr("dataset_factory", dataset_factory)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun paddedBatchDataset(input_dataset: Tensor, batch_size: Tensor, padded_shapes: Array<Tensor>, padding_values: Tensor, output_shapes: Array<Dimension>, name: String = "PaddedBatchDataset") = run {
    tf.buildOpTensor("PaddedBatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(padded_shapes, false)
      addInput(padding_values, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun parallelInterleaveDataset(input_dataset: Tensor, other_arguments: Tensor, cycle_length: Tensor, block_length: Tensor, sloppy: Tensor, buffer_output_elements: Tensor, prefetch_input_elements: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "ParallelInterleaveDataset") = run {
    tf.buildOpTensor("ParallelInterleaveDataset", name) {
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
  
  fun parallelMapDataset(input_dataset: Tensor, other_arguments: Tensor, num_parallel_calls: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "ParallelMapDataset") = run {
    tf.buildOpTensor("ParallelMapDataset", name) {
      addInput(input_dataset, false)
      addInput(other_arguments, false)
      addInput(num_parallel_calls, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun prefetchDataset(input_dataset: Tensor, buffer_size: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "PrefetchDataset") = run {
    tf.buildOpTensor("PrefetchDataset", name) {
      addInput(input_dataset, false)
      addInput(buffer_size, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun prependFromQueueAndPaddedBatchDataset(input_dataset: Tensor, batch_size: Tensor, padded_shapes: Array<Tensor>, padding_values: Tensor, output_shapes: Array<Dimension>, name: String = "PrependFromQueueAndPaddedBatchDataset") = run {
    tf.buildOpTensor("PrependFromQueueAndPaddedBatchDataset", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(padded_shapes, false)
      addInput(padding_values, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun randomDataset(seed: Tensor, seed2: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "RandomDataset") = run {
    tf.buildOpTensor("RandomDataset", name) {
      addInput(seed, false)
      addInput(seed2, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun rangeDataset(start: Tensor, stop: Tensor, step: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "RangeDataset") = run {
    tf.buildOpTensor("RangeDataset", name) {
      addInput(start, false)
      addInput(stop, false)
      addInput(step, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun repeatDataset(input_dataset: Tensor, count: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "RepeatDataset") = run {
    tf.buildOpTensor("RepeatDataset", name) {
      addInput(input_dataset, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun scanDataset(input_dataset: Tensor, initial_state: Tensor, other_arguments: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "ScanDataset") = run {
    tf.buildOpTensor("ScanDataset", name) {
      addInput(input_dataset, false)
      addInput(initial_state, false)
      addInput(other_arguments, false)
      attr("f", f)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun serializeIterator(resource_handle: Tensor, name: String = "SerializeIterator") = run {
    tf.buildOpTensor("SerializeIterator", name) {
      addInput(resource_handle, false)
    }
  }
  
  fun setStatsAggregatorDataset(input_dataset: Tensor, stats_aggregator: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "SetStatsAggregatorDataset") = run {
    tf.buildOpTensor("SetStatsAggregatorDataset", name) {
      addInput(input_dataset, false)
      addInput(stats_aggregator, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun shuffleAndRepeatDataset(input_dataset: Tensor, buffer_size: Tensor, seed: Tensor, seed2: Tensor, count: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "ShuffleAndRepeatDataset") = run {
    tf.buildOpTensor("ShuffleAndRepeatDataset", name) {
      addInput(input_dataset, false)
      addInput(buffer_size, false)
      addInput(seed, false)
      addInput(seed2, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun shuffleDataset(input_dataset: Tensor, buffer_size: Tensor, seed: Tensor, seed2: Tensor, reshuffle_each_iteration: Boolean = true, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "ShuffleDataset") = run {
    tf.buildOpTensor("ShuffleDataset", name) {
      addInput(input_dataset, false)
      addInput(buffer_size, false)
      addInput(seed, false)
      addInput(seed2, false)
      attr("reshuffle_each_iteration", reshuffle_each_iteration)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun skipDataset(input_dataset: Tensor, count: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "SkipDataset") = run {
    tf.buildOpTensor("SkipDataset", name) {
      addInput(input_dataset, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun slideDataset(input_dataset: Tensor, window_size: Tensor, window_shift: Tensor, window_stride: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "SlideDataset") = run {
    tf.buildOpTensor("SlideDataset", name) {
      addInput(input_dataset, false)
      addInput(window_size, false)
      addInput(window_shift, false)
      addInput(window_stride, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun sparseTensorSliceDataset(indices: Tensor, values: Tensor, dense_shape: Tensor, name: String = "SparseTensorSliceDataset") = run {
    tf.buildOpTensor("SparseTensorSliceDataset", name) {
      addInput(indices, false)
      addInput(values, false)
      addInput(dense_shape, false)
    }
  }
  
  fun sqlDataset(driver_name: Tensor, data_source_name: Tensor, query: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "SqlDataset") = run {
    tf.buildOpTensor("SqlDataset", name) {
      addInput(driver_name, false)
      addInput(data_source_name, false)
      addInput(query, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun statsAggregatorHandle(container: String = "", shared_name: String = "", name: String = "StatsAggregatorHandle") = run {
    tf.buildOpTensor("StatsAggregatorHandle", name) {
      attr("container", container)
      attr("shared_name", shared_name)
    }
  }
  
  fun statsAggregatorSummary(iterator: Tensor, name: String = "StatsAggregatorSummary") = run {
    tf.buildOpTensor("StatsAggregatorSummary", name) {
      addInput(iterator, false)
    }
  }
  
  fun tFRecordDataset(filenames: Tensor, compression_type: Tensor, buffer_size: Tensor, name: String = "TFRecordDataset") = run {
    tf.buildOpTensor("TFRecordDataset", name) {
      addInput(filenames, false)
      addInput(compression_type, false)
      addInput(buffer_size, false)
    }
  }
  
  fun takeDataset(input_dataset: Tensor, count: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "TakeDataset") = run {
    tf.buildOpTensor("TakeDataset", name) {
      addInput(input_dataset, false)
      addInput(count, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun tensorDataset(components: Tensor, output_shapes: Array<Dimension>, name: String = "TensorDataset") = run {
    tf.buildOpTensor("TensorDataset", name) {
      addInput(components, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun tensorSliceDataset(components: Tensor, output_shapes: Array<Dimension>, name: String = "TensorSliceDataset") = run {
    tf.buildOpTensor("TensorSliceDataset", name) {
      addInput(components, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun textLineDataset(filenames: Tensor, compression_type: Tensor, buffer_size: Tensor, name: String = "TextLineDataset") = run {
    tf.buildOpTensor("TextLineDataset", name) {
      addInput(filenames, false)
      addInput(compression_type, false)
      addInput(buffer_size, false)
    }
  }
  
  fun unbatchDataset(input_dataset: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "UnbatchDataset") = run {
    tf.buildOpTensor("UnbatchDataset", name) {
      addInput(input_dataset, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun zipDataset(input_datasets: Array<Tensor>, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "ZipDataset") = run {
    tf.buildOpTensor("ZipDataset", name) {
      addInput(input_datasets, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun batchDatasetV2(input_dataset: Tensor, batch_size: Tensor, drop_remainder: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "BatchDatasetV2") = run {
    tf.buildOpTensor("BatchDatasetV2", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(drop_remainder, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun datasetToGraph(input_dataset: Tensor, name: String = "DatasetToGraph") = run {
    tf.buildOpTensor("DatasetToGraph", name) {
      addInput(input_dataset, false)
    }
  }
  
  fun datasetToTFRecord(input_dataset: Tensor, filename: Tensor, compression_type: Tensor, name: String = "DatasetToTFRecord") = run {
    tf.buildOp("DatasetToTFRecord", name) {
      addInput(input_dataset, false)
      addInput(filename, false)
      addInput(compression_type, false)
    }
  }
  
  fun groupByReducerDataset(input_dataset: Tensor, key_func_other_arguments: Tensor, init_func_other_arguments: Tensor, reduce_func_other_arguments: Tensor, finalize_func_other_arguments: Tensor, key_func: NameAttrList, init_func: NameAttrList, reduce_func: NameAttrList, finalize_func: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "GroupByReducerDataset") = run {
    tf.buildOpTensor("GroupByReducerDataset", name) {
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
  
  fun iteratorFromStringHandleV2(string_handle: Tensor, output_types: Array<Long> = arrayOf(), output_shapes: Array<Dimension> = arrayOf(), name: String = "IteratorFromStringHandleV2") = run {
    tf.buildOpTensor("IteratorFromStringHandleV2", name) {
      addInput(string_handle, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun iteratorV2(shared_name: String, container: String, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "IteratorV2") = run {
    tf.buildOpTensor("IteratorV2", name) {
      attr("shared_name", shared_name)
      attr("container", container)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun mapAndBatchDataset(input_dataset: Tensor, other_arguments: Tensor, batch_size: Tensor, num_parallel_batches: Tensor, drop_remainder: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "MapAndBatchDataset") = run {
    tf.buildOpTensor("MapAndBatchDataset", name) {
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
  
  fun mapAndBatchDatasetV2(input_dataset: Tensor, other_arguments: Tensor, batch_size: Tensor, num_parallel_calls: Tensor, drop_remainder: Tensor, f: NameAttrList, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "MapAndBatchDatasetV2") = run {
    tf.buildOpTensor("MapAndBatchDatasetV2", name) {
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
  
  fun optimizeDataset(input_dataset: Tensor, optimizations: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "OptimizeDataset") = run {
    tf.buildOpTensor("OptimizeDataset", name) {
      addInput(input_dataset, false)
      addInput(optimizations, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun paddedBatchDatasetV2(input_dataset: Tensor, batch_size: Tensor, padded_shapes: Array<Tensor>, padding_values: Tensor, drop_remainder: Tensor, output_shapes: Array<Dimension>, name: String = "PaddedBatchDatasetV2") = run {
    tf.buildOpTensor("PaddedBatchDatasetV2", name) {
      addInput(input_dataset, false)
      addInput(batch_size, false)
      addInput(padded_shapes, false)
      addInput(padding_values, false)
      addInput(drop_remainder, false)
      attr("output_shapes", output_shapes)
    }
  }
  
  fun sinkDataset(input_dataset: Tensor, name: String = "SinkDataset") = run {
    tf.buildOpTensor("SinkDataset", name) {
      addInput(input_dataset, false)
    }
  }
  
  fun windowDataset(input_dataset: Tensor, window_size: Tensor, output_types: Array<Long>, output_shapes: Array<Dimension>, name: String = "WindowDataset") = run {
    tf.buildOpTensor("WindowDataset", name) {
      addInput(input_dataset, false)
      addInput(window_size, false)
      attr("output_types", output_types)
      attr("output_shapes", output_shapes)
    }
  }
}