package wumo.sim.util

import java.io.File
import java.io.File.separatorChar
import java.nio.file.Files.copy
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

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

fun unpackDirToTemp(resourceDir: String, override: Boolean = false): String {
  val loader = Thread.currentThread().contextClassLoader
  val dir = File(loader.getResource(resourceDir).path)
  dir.listFiles().forEach {
    unpackFileToTemp(Paths.get(resourceDir, it.name).toString(),
                     override)
  }
  return Paths.get(System.getProperty("java.io.tmpdir"), resourceDir).toString()
}

fun listResources(resourceDir: String): Array<File> {
  val loader = Thread.currentThread().contextClassLoader
  val dir = File(loader.getResource(resourceDir).path)
  return dir.listFiles()
}