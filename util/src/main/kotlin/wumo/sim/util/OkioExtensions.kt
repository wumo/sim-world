package wumo.sim.util

import okio.*
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

fun InputStream.source(): Source = Okio.source(this)
fun File.sink(append: Boolean = false): Sink = Okio.sink(FileOutputStream(this, append))
fun File.source(): Source = inputStream().source()
fun Source.buffer(): BufferedSource = Okio.buffer(this)
fun Sink.buffer(): BufferedSink = Okio.buffer(this)