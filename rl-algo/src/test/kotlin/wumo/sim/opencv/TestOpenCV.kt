package wumo.sim.opencv

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgcodecs.imwrite
import org.bytedeco.javacpp.opencv_imgproc.*
import org.junit.Assert
import org.junit.Test
import wumo.sim.algorithm.drl.common.toMat
import wumo.sim.algorithm.drl.common.toNDArray
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray

class TestOpenCV {
  @Test
  fun TestMat() {
    val src = Mat(160, 160, CV_8UC(3))
    
    for (i in 0 until src.rows()) {
      val row = src.ptr(i)
      var k = 0L
      for (j in 0 until src.cols()) {
        row.position(k++).put(0)//blue
        row.position(k++).put(0)//green
        row.position(k++).put(255.toByte())//red
      }
    }
    for (i in 0 until src.rows()) {
      val row = src.ptr(i)
      var k = 0L
      for (j in 0 until src.cols())
        for (l in 0 until 3) {
          println(row[k])
          k++
        }
    }
//        row.position(k)
//        row.put(127.toByte())
//        row.position(k + 1L)
//        row.put(125.toByte())
//        row.position(k + 2L)
//        row.put(77.toByte())
//        k += 2
    imwrite("test.jpg", src)
  }
  
  @Test
  fun TestMat2() {
    val data = ByteArray(160 * 160 * 3) {
      if (it % 3 == 2) 255.toByte()
      else 0
    }
    val src = Mat(160, 160, CV_8UC(3), BytePointer(*data))

//    for (i in 0 until src.rows()) {
//      val row = src.ptr(i)
//      var k = 0L
//      for (j in 0 until src.cols()) {
//        row.position(k++).put(0)//blue
//        row.position(k++).put(0)//green
//        row.position(k++).put(255.toByte())//red
//      }
//    }
    for (i in 0 until src.rows()) {
      val row = src.ptr(i)
      var k = 0L
      for (j in 0 until src.cols())
        for (l in 0 until 3) {
          println(row[k])
          k++
        }
    }
//        row.position(k)
//        row.put(127.toByte())
//        row.position(k + 1L)
//        row.put(125.toByte())
//        row.position(k + 2L)
//        row.put(77.toByte())
//        k += 2
    imwrite("test2.jpg", src)
  }
  
  @Test
  fun cvtColorTest() {
    val src = imread("origin.jpg")
    val dst = Mat()
    cvtColor(src, dst, COLOR_RGB2GRAY)
    imwrite("test3.jpg", dst)
  }
  
  @Test
  fun resizeTest() {
    val src = imread("origin.jpg")
    println(src.type())
    println(CV_8UC3)
    val dst = Mat()
    cvtColor(src, dst, COLOR_RGB2GRAY)
    println(dst.type())
    println(CV_8UC1)
    val dst2 = Mat()
    resize(dst, dst2, Size(84, 84), 0.0, 0.0, INTER_AREA)
    imwrite("test4.jpg", dst2)
  }
  
  @Test
  fun ndarrayTest() {
    val data = ByteArray(160 * 160 * 3) {
      if (it % 3 == 2) 255.toByte()
      else 0
    }
    val ndarray1 = NDArray(Shape(160, 160, 3), data)
    val mat = ndarray1.toMat()
    imwrite("test5.jpg", mat)
    val ndarray2 = mat.toNDArray()
    Assert.assertEquals(ndarray1, ndarray2)
  }
}