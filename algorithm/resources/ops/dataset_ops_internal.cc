// This file is MACHINE GENERATED! Do not edit.


#include "tensorflow/cc/ops/const_op.h"
#include "G:/vs_projects/tensorflow/tensorflow/contrib/cmake/build/tensorflow/cc/ops/dataset_ops_internal.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

BatchDatasetV2::BatchDatasetV2(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input_dataset,
                               ::tensorflow::Input batch_size,
                               ::tensorflow::Input drop_remainder, const
                               DataTypeSlice& output_types, const
                               gtl::ArraySlice<PartialTensorShape>&
                               output_shapes) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _batch_size = ::tensorflow::ops::AsNodeOut(scope, batch_size);
  if (!scope.ok()) return;
  auto _drop_remainder = ::tensorflow::ops::AsNodeOut(scope, drop_remainder);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("BatchDatasetV2");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "BatchDatasetV2")
                     .Input(_input_dataset)
                     .Input(_batch_size)
                     .Input(_drop_remainder)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

DatasetToGraph::DatasetToGraph(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input input_dataset) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("DatasetToGraph");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "DatasetToGraph")
                     .Input(_input_dataset)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->graph = Output(ret, 0);
}

DatasetToTFRecord::DatasetToTFRecord(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input input_dataset,
                                     ::tensorflow::Input filename,
                                     ::tensorflow::Input compression_type) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _filename = ::tensorflow::ops::AsNodeOut(scope, filename);
  if (!scope.ok()) return;
  auto _compression_type = ::tensorflow::ops::AsNodeOut(scope, compression_type);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("DatasetToTFRecord");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "DatasetToTFRecord")
                     .Input(_input_dataset)
                     .Input(_filename)
                     .Input(_compression_type)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->operation = Operation(ret);
  return;
}

GroupByReducerDataset::GroupByReducerDataset(const ::tensorflow::Scope& scope,
                                             ::tensorflow::Input input_dataset,
                                             ::tensorflow::InputList
                                             key_func_other_arguments,
                                             ::tensorflow::InputList
                                             init_func_other_arguments,
                                             ::tensorflow::InputList
                                             reduce_func_other_arguments,
                                             ::tensorflow::InputList
                                             finalize_func_other_arguments,
                                             const NameAttrList& key_func,
                                             const NameAttrList& init_func,
                                             const NameAttrList& reduce_func,
                                             const NameAttrList& finalize_func,
                                             const DataTypeSlice& output_types,
                                             const
                                             gtl::ArraySlice<PartialTensorShape>&
                                             output_shapes) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _key_func_other_arguments = ::tensorflow::ops::AsNodeOutList(scope, key_func_other_arguments);
  if (!scope.ok()) return;
  auto _init_func_other_arguments = ::tensorflow::ops::AsNodeOutList(scope, init_func_other_arguments);
  if (!scope.ok()) return;
  auto _reduce_func_other_arguments = ::tensorflow::ops::AsNodeOutList(scope, reduce_func_other_arguments);
  if (!scope.ok()) return;
  auto _finalize_func_other_arguments = ::tensorflow::ops::AsNodeOutList(scope, finalize_func_other_arguments);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("GroupByReducerDataset");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "GroupByReducerDataset")
                     .Input(_input_dataset)
                     .Input(_key_func_other_arguments)
                     .Input(_init_func_other_arguments)
                     .Input(_reduce_func_other_arguments)
                     .Input(_finalize_func_other_arguments)
                     .Attr("key_func", key_func)
                     .Attr("init_func", init_func)
                     .Attr("reduce_func", reduce_func)
                     .Attr("finalize_func", finalize_func)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

IteratorFromStringHandleV2::IteratorFromStringHandleV2(const
                                                       ::tensorflow::Scope&
                                                       scope,
                                                       ::tensorflow::Input
                                                       string_handle, const
                                                       IteratorFromStringHandleV2::Attrs&
                                                       attrs) {
  if (!scope.ok()) return;
  auto _string_handle = ::tensorflow::ops::AsNodeOut(scope, string_handle);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("IteratorFromStringHandleV2");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "IteratorFromStringHandleV2")
                     .Input(_string_handle)
                     .Attr("output_types", attrs.output_types_)
                     .Attr("output_shapes", attrs.output_shapes_)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->resource_handle = Output(ret, 0);
}

IteratorFromStringHandleV2::IteratorFromStringHandleV2(const
                                                       ::tensorflow::Scope&
                                                       scope,
                                                       ::tensorflow::Input
                                                       string_handle)
  : IteratorFromStringHandleV2(scope, string_handle, IteratorFromStringHandleV2::Attrs()) {}

IteratorV2::IteratorV2(const ::tensorflow::Scope& scope, StringPiece
                       shared_name, StringPiece container, const DataTypeSlice&
                       output_types, const gtl::ArraySlice<PartialTensorShape>&
                       output_shapes) {
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("IteratorV2");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "IteratorV2")
                     .Attr("shared_name", shared_name)
                     .Attr("container", container)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

MapAndBatchDataset::MapAndBatchDataset(const ::tensorflow::Scope& scope,
                                       ::tensorflow::Input input_dataset,
                                       ::tensorflow::InputList other_arguments,
                                       ::tensorflow::Input batch_size,
                                       ::tensorflow::Input
                                       num_parallel_batches,
                                       ::tensorflow::Input drop_remainder,
                                       const NameAttrList& f, const
                                       DataTypeSlice& output_types, const
                                       gtl::ArraySlice<PartialTensorShape>&
                                       output_shapes) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _other_arguments = ::tensorflow::ops::AsNodeOutList(scope, other_arguments);
  if (!scope.ok()) return;
  auto _batch_size = ::tensorflow::ops::AsNodeOut(scope, batch_size);
  if (!scope.ok()) return;
  auto _num_parallel_batches = ::tensorflow::ops::AsNodeOut(scope, num_parallel_batches);
  if (!scope.ok()) return;
  auto _drop_remainder = ::tensorflow::ops::AsNodeOut(scope, drop_remainder);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("MapAndBatchDataset");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "MapAndBatchDataset")
                     .Input(_input_dataset)
                     .Input(_other_arguments)
                     .Input(_batch_size)
                     .Input(_num_parallel_batches)
                     .Input(_drop_remainder)
                     .Attr("f", f)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

MapAndBatchDatasetV2::MapAndBatchDatasetV2(const ::tensorflow::Scope& scope,
                                           ::tensorflow::Input input_dataset,
                                           ::tensorflow::InputList
                                           other_arguments, ::tensorflow::Input
                                           batch_size, ::tensorflow::Input
                                           num_parallel_calls,
                                           ::tensorflow::Input drop_remainder,
                                           const NameAttrList& f, const
                                           DataTypeSlice& output_types, const
                                           gtl::ArraySlice<PartialTensorShape>&
                                           output_shapes) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _other_arguments = ::tensorflow::ops::AsNodeOutList(scope, other_arguments);
  if (!scope.ok()) return;
  auto _batch_size = ::tensorflow::ops::AsNodeOut(scope, batch_size);
  if (!scope.ok()) return;
  auto _num_parallel_calls = ::tensorflow::ops::AsNodeOut(scope, num_parallel_calls);
  if (!scope.ok()) return;
  auto _drop_remainder = ::tensorflow::ops::AsNodeOut(scope, drop_remainder);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("MapAndBatchDatasetV2");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "MapAndBatchDatasetV2")
                     .Input(_input_dataset)
                     .Input(_other_arguments)
                     .Input(_batch_size)
                     .Input(_num_parallel_calls)
                     .Input(_drop_remainder)
                     .Attr("f", f)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

OptimizeDataset::OptimizeDataset(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input input_dataset,
                                 ::tensorflow::Input optimizations, const
                                 DataTypeSlice& output_types, const
                                 gtl::ArraySlice<PartialTensorShape>&
                                 output_shapes) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _optimizations = ::tensorflow::ops::AsNodeOut(scope, optimizations);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("OptimizeDataset");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "OptimizeDataset")
                     .Input(_input_dataset)
                     .Input(_optimizations)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

PaddedBatchDatasetV2::PaddedBatchDatasetV2(const ::tensorflow::Scope& scope,
                                           ::tensorflow::Input input_dataset,
                                           ::tensorflow::Input batch_size,
                                           ::tensorflow::InputList
                                           padded_shapes,
                                           ::tensorflow::InputList
                                           padding_values, ::tensorflow::Input
                                           drop_remainder, const
                                           gtl::ArraySlice<PartialTensorShape>&
                                           output_shapes) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _batch_size = ::tensorflow::ops::AsNodeOut(scope, batch_size);
  if (!scope.ok()) return;
  auto _padded_shapes = ::tensorflow::ops::AsNodeOutList(scope, padded_shapes);
  if (!scope.ok()) return;
  auto _padding_values = ::tensorflow::ops::AsNodeOutList(scope, padding_values);
  if (!scope.ok()) return;
  auto _drop_remainder = ::tensorflow::ops::AsNodeOut(scope, drop_remainder);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("PaddedBatchDatasetV2");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "PaddedBatchDatasetV2")
                     .Input(_input_dataset)
                     .Input(_batch_size)
                     .Input(_padded_shapes)
                     .Input(_padding_values)
                     .Input(_drop_remainder)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

SinkDataset::SinkDataset(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input_dataset) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("SinkDataset");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "SinkDataset")
                     .Input(_input_dataset)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

WindowDataset::WindowDataset(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input input_dataset,
                             ::tensorflow::Input window_size, const
                             DataTypeSlice& output_types, const
                             gtl::ArraySlice<PartialTensorShape>&
                             output_shapes) {
  if (!scope.ok()) return;
  auto _input_dataset = ::tensorflow::ops::AsNodeOut(scope, input_dataset);
  if (!scope.ok()) return;
  auto _window_size = ::tensorflow::ops::AsNodeOut(scope, window_size);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("WindowDataset");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "WindowDataset")
                     .Input(_input_dataset)
                     .Input(_window_size)
                     .Attr("output_types", output_types)
                     .Attr("output_shapes", output_shapes)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  scope.UpdateStatus(scope.DoShapeInference(ret));
  this->handle = Output(ret, 0);
}

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow
