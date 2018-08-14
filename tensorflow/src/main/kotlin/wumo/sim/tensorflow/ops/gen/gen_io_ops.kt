/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.ops.Output
import wumo.sim.util.Shape
import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.tf
import wumo.sim.util.ndarray.NDArray

interface gen_io_ops {
fun _fixedLengthRecordReaderV2(record_bytes: Long, header_bytes: Long = 0L, footer_bytes: Long = 0L, hop_bytes: Long = 0L, container: String = "", shared_name: String = "", encoding: String = "", name: String = "FixedLengthRecordReaderV2") = run {
buildOpTensor("FixedLengthRecordReaderV2", name){
attr("record_bytes", record_bytes)
attr("header_bytes", header_bytes)
attr("footer_bytes", footer_bytes)
attr("hop_bytes", hop_bytes)
attr("container", container)
attr("shared_name", shared_name)
attr("encoding", encoding)
}
}
fun _identityReaderV2(container: String = "", shared_name: String = "", name: String = "IdentityReaderV2") = run {
buildOpTensor("IdentityReaderV2", name){
attr("container", container)
attr("shared_name", shared_name)
}
}
fun _lMDBReader(container: String = "", shared_name: String = "", name: String = "LMDBReader") = run {
buildOpTensor("LMDBReader", name){
attr("container", container)
attr("shared_name", shared_name)
}
}
fun _matchingFiles(pattern: Output, name: String = "MatchingFiles") = run {
buildOpTensor("MatchingFiles", name){
addInput(pattern,false)
}
}
fun _mergeV2Checkpoints(checkpoint_prefixes: Output, destination_prefix: Output, delete_old_dirs: Boolean = true, name: String = "MergeV2Checkpoints") = run {
buildOp("MergeV2Checkpoints", name){
addInput(checkpoint_prefixes,false)
addInput(destination_prefix,false)
attr("delete_old_dirs", delete_old_dirs)
}
}
fun _readFile(filename: Output, name: String = "ReadFile") = run {
buildOpTensor("ReadFile", name){
addInput(filename,false)
}
}
fun _readerNumRecordsProducedV2(reader_handle: Output, name: String = "ReaderNumRecordsProducedV2") = run {
buildOpTensor("ReaderNumRecordsProducedV2", name){
addInput(reader_handle,false)
}
}
fun _readerNumWorkUnitsCompletedV2(reader_handle: Output, name: String = "ReaderNumWorkUnitsCompletedV2") = run {
buildOpTensor("ReaderNumWorkUnitsCompletedV2", name){
addInput(reader_handle,false)
}
}
fun _readerReadUpToV2(reader_handle: Output, queue_handle: Output, num_records: Output, name: String = "ReaderReadUpToV2") = run {
buildOpTensors("ReaderReadUpToV2", name){
addInput(reader_handle,false)
addInput(queue_handle,false)
addInput(num_records,false)
}
}
fun _readerReadV2(reader_handle: Output, queue_handle: Output, name: String = "ReaderReadV2") = run {
buildOpTensors("ReaderReadV2", name){
addInput(reader_handle,false)
addInput(queue_handle,false)
}
}
fun _readerResetV2(reader_handle: Output, name: String = "ReaderResetV2") = run {
buildOp("ReaderResetV2", name){
addInput(reader_handle,false)
}
}
fun _readerRestoreStateV2(reader_handle: Output, state: Output, name: String = "ReaderRestoreStateV2") = run {
buildOp("ReaderRestoreStateV2", name){
addInput(reader_handle,false)
addInput(state,false)
}
}
fun _readerSerializeStateV2(reader_handle: Output, name: String = "ReaderSerializeStateV2") = run {
buildOpTensor("ReaderSerializeStateV2", name){
addInput(reader_handle,false)
}
}
fun _restore(file_pattern: Output, tensor_name: Output, dt: Int, preferred_shard: Long = -1L, name: String = "Restore") = run {
buildOpTensor("Restore", name){
addInput(file_pattern,false)
addInput(tensor_name,false)
attrType("dt", dt)
attr("preferred_shard", preferred_shard)
}
}
fun _restoreSlice(file_pattern: Output, tensor_name: Output, shape_and_slice: Output, dt: Int, preferred_shard: Long = -1L, name: String = "RestoreSlice") = run {
buildOpTensor("RestoreSlice", name){
addInput(file_pattern,false)
addInput(tensor_name,false)
addInput(shape_and_slice,false)
attrType("dt", dt)
attr("preferred_shard", preferred_shard)
}
}
fun _restoreV2(prefix: Output, tensor_names: Output, shape_and_slices: Output, dtypes: Array<Long>, name: String = "RestoreV2") = run {
buildOpTensors("RestoreV2", name){
addInput(prefix,false)
addInput(tensor_names,false)
addInput(shape_and_slices,false)
attr("dtypes", dtypes)
}
}
fun _save(filename: Output, tensor_names: Output, data: Output, name: String = "Save") = run {
buildOp("Save", name){
addInput(filename,false)
addInput(tensor_names,false)
addInput(data,false)
}
}
fun _saveSlices(filename: Output, tensor_names: Output, shapes_and_slices: Output, data: Output, name: String = "SaveSlices") = run {
buildOp("SaveSlices", name){
addInput(filename,false)
addInput(tensor_names,false)
addInput(shapes_and_slices,false)
addInput(data,false)
}
}
fun _saveV2(prefix: Output, tensor_names: Output, shape_and_slices: Output, tensors: Output, name: String = "SaveV2") = run {
buildOp("SaveV2", name){
addInput(prefix,false)
addInput(tensor_names,false)
addInput(shape_and_slices,false)
addInput(tensors,false)
}
}
fun _shardedFilename(basename: Output, shard: Output, num_shards: Output, name: String = "ShardedFilename") = run {
buildOpTensor("ShardedFilename", name){
addInput(basename,false)
addInput(shard,false)
addInput(num_shards,false)
}
}
fun _shardedFilespec(basename: Output, num_shards: Output, name: String = "ShardedFilespec") = run {
buildOpTensor("ShardedFilespec", name){
addInput(basename,false)
addInput(num_shards,false)
}
}
fun _tFRecordReaderV2(container: String = "", shared_name: String = "", compression_type: String = "", name: String = "TFRecordReaderV2") = run {
buildOpTensor("TFRecordReaderV2", name){
attr("container", container)
attr("shared_name", shared_name)
attr("compression_type", compression_type)
}
}
fun _textLineReaderV2(skip_header_lines: Long = 0L, container: String = "", shared_name: String = "", name: String = "TextLineReaderV2") = run {
buildOpTensor("TextLineReaderV2", name){
attr("skip_header_lines", skip_header_lines)
attr("container", container)
attr("shared_name", shared_name)
}
}
fun _wholeFileReaderV2(container: String = "", shared_name: String = "", name: String = "WholeFileReaderV2") = run {
buildOpTensor("WholeFileReaderV2", name){
attr("container", container)
attr("shared_name", shared_name)
}
}
fun _writeFile(filename: Output, contents: Output, name: String = "WriteFile") = run {
buildOp("WriteFile", name){
addInput(filename,false)
addInput(contents,false)
}
}
}