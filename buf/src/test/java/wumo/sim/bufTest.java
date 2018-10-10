package wumo.sim;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteOrder;

import static org.junit.Assert.*;

public class bufTest {
  
  @Before
  public void startup() {
    buf.buf_init();
  }
  
  @Test
  public void maxOf() {
    long size = 100;
    BytePointer a = new BytePointer(size);
    for (long i = 0; i < size; i++) {
      a.put(i, (byte) (i % 128));
    }
    BytePointer b = new BytePointer(size);
    for (long i = 0; i < size; i++) {
      b.put(i, (byte) (i + 1 % 128));
    }
    BytePointer c = new BytePointer(size);
    buf.maxOf(a, b, c, size);
  }
  
  @Test
  public void copy() {
    long size = 1000;
    BytePointer a = new BytePointer(size);
    for (long i = 0; i < size; i++) {
      a.put(i, (byte) i);
    }
    FloatPointer b = new FloatPointer(size);
    buf.castchar2float(a, b, size);
    for (long i = 0; i < size; i++) {
      System.out.println(a.get(i) + "," + b.get(i));
    }
    for (long i = 0; i < size; i++) {
      b.put(i, i);
    }
    buf.castfloat2char(b, a, size);
    for (long i = 0; i < size; i++) {
      System.out.println(a.get(i) + "," + b.get(i));
    }
  }
  
  @Test
  public void testCopy() {
    BytePointer a = new BytePointer(128);
    BytePointer b = new BytePointer(4);
    b.putInt(8);
    Pointer.memcpy(a.position(0L), b.position(0L), 4);
    System.out.println(a.getInt(0L));
    Pointer.memcpy(a.position(4L), b.position(0L), 4);
    a.position(0L);
    System.out.println(a.getInt(4L));
    Pointer.memcpy(a.position(8L), b.position(0L), 4);
    a.position(0L);
    System.out.println(a.getInt(8L));
    Pointer.memcpy(a.position(12L), b.position(0L), 4);
    a.position(0L);
    System.out.println(a.getInt(12L));
  }
}