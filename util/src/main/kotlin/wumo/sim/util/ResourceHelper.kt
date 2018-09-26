package wumo.sim.util

import java.nio.file.*
import java.nio.file.Files.copy

fun unpackFileToTemp(resource: String, override: Boolean = false): String {
  val path = Paths.get(System.getProperty("java.io.tmpdir"), resource)
  val file = path.toFile()
  if (!override && file.exists()) return file.path
  file.parentFile.mkdirs()
  val source = Thread.currentThread().contextClassLoader
      .getResourceAsStream(resource)
  copy(source, path, StandardCopyOption.REPLACE_EXISTING)
  return file.path
}

fun unpackDirToTemp(resourceDir: String, override: Boolean = false)
    : Pair<String, List<String>> {
  val uri = Thread.currentThread().contextClassLoader
      .getResource(resourceDir).toURI()
  val myPath = if (uri.scheme == "jar") {
    val fs = FileSystems.newFileSystem(uri, emptyMap<String,Any>())
    fs.getPath(resourceDir)
  } else {
    Paths.get(uri)
  }
  val files = mutableListOf<String>()
  Files.list(myPath).forEach {
    unpackFileToTemp(Paths.get(resourceDir, it.fileName.toString()).toString(), override)
    files += it.fileName.toString()
  }
  return Pair(Paths.get(System.getProperty("java.io.tmpdir"), resourceDir).toString(),
              files)
}