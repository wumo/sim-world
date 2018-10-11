package wumo.sim;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.*;

public class bufTest {
  
  @Before
  public void startup() {
    buf.buf_init();
  }
  
  public void print(FloatPointer a) {
    for (int i = 0; i < a.limit(); i++) {
      System.out.println(a.get(i));
    }
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
  public void castTest() {
    long size = 1000;
    BytePointer a = new BytePointer(size);
    for (long i = 0; i < size; i++) {
      a.put(i, (byte) i);
    }
    FloatPointer b = new FloatPointer(size);
    buf.castunsignedchar2float(a, b, size);
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
  
  @Test
  public void testAbs() {
    FloatPointer a = new FloatPointer(128);
    FloatPointer b = new FloatPointer(128);
    for (int i = 0; i < 128; i++) {
      a.put(i, (float) (Math.signum(Math.random() - 0.5) * Math.random()));
      b.put(i, a.get(i));
    }
    buf.absf(b, b.limit());
    for (int i = 0; i < 128; i++) {
      assertEquals(b.get(i), Math.abs(a.get(i)), 1e-9);
    }
  }
  
  @Test
  public void testAdd() {
    FloatPointer a = new FloatPointer(128);
    FloatPointer b = new FloatPointer(128);
    for (int i = 0; i < 128; i++) {
      a.put(i, (float) (Math.signum(Math.random() - 0.5) * Math.random()));
      b.put(i, a.get(i));
    }
    buf.addf(b, b.limit(), 2);
    for (int i = 0; i < 128; i++) {
      assertEquals(b.get(i), a.get(i) + 2, 1e-9);
    }
  }
}