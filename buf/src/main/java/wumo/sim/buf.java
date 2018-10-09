// Targeted by JavaCPP version 1.4.3-SNAPSHOT: DO NOT EDIT THIS FILE

package wumo.sim;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class buf extends wumo.sim.presets.buf {
    static { Loader.load(); }

// Parsed from buf/buf.h

// #ifndef BUF_LIBRARY_H
// #define BUF_LIBRARY_H

// #include <cstddef>
// #include <algorithm>

public static native void buf_init();

public static native void maxOf(@Cast("char*") BytePointer buf_a, @Cast("char*") BytePointer buf_b, @Cast("char*") BytePointer output, @Cast("size_t") long size);
public static native void maxOf(@Cast("char*") ByteBuffer buf_a, @Cast("char*") ByteBuffer buf_b, @Cast("char*") ByteBuffer output, @Cast("size_t") long size);
public static native void maxOf(@Cast("char*") byte[] buf_a, @Cast("char*") byte[] buf_b, @Cast("char*") byte[] output, @Cast("size_t") long size);

public static native void concat(int axis, @Cast("unsigned char**") PointerPointer array, @Cast("int**") PointerPointer shape, @Cast("size_t") long size,
            @Cast("size_t") long byteSize, @Cast("unsigned char*") BytePointer output, IntPointer result_shape);
public static native void concat(int axis, @Cast("unsigned char**") @ByPtrPtr BytePointer array, @ByPtrPtr IntPointer shape, @Cast("size_t") long size,
            @Cast("size_t") long byteSize, @Cast("unsigned char*") BytePointer output, IntPointer result_shape);
public static native void concat(int axis, @Cast("unsigned char**") @ByPtrPtr ByteBuffer array, @ByPtrPtr IntBuffer shape, @Cast("size_t") long size,
            @Cast("size_t") long byteSize, @Cast("unsigned char*") ByteBuffer output, IntBuffer result_shape);
public static native void concat(int axis, @Cast("unsigned char**") @ByPtrPtr byte[] array, @ByPtrPtr int[] shape, @Cast("size_t") long size,
            @Cast("size_t") long byteSize, @Cast("unsigned char*") byte[] output, int[] result_shape);

public static native void divf(FloatPointer buf, @Cast("size_t") long size, float s);
public static native void divf(FloatBuffer buf, @Cast("size_t") long size, float s);
public static native void divf(float[] buf, @Cast("size_t") long size, float s);

public static native @Name("cast<unsigned char,unsigned char>") void castunsignedchar2unsignedchar(@Cast("unsigned char*") BytePointer a, @Cast("unsigned char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,unsigned char>") void castunsignedchar2unsignedchar(@Cast("unsigned char*") ByteBuffer a, @Cast("unsigned char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,unsigned char>") void castunsignedchar2unsignedchar(@Cast("unsigned char*") byte[] a, @Cast("unsigned char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<unsigned char,char>") void castunsignedchar2char(@Cast("unsigned char*") BytePointer a, @Cast("char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,char>") void castunsignedchar2char(@Cast("unsigned char*") ByteBuffer a, @Cast("char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,char>") void castunsignedchar2char(@Cast("unsigned char*") byte[] a, @Cast("char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<unsigned char,short>") void castunsignedchar2short(@Cast("unsigned char*") BytePointer a, ShortPointer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,short>") void castunsignedchar2short(@Cast("unsigned char*") ByteBuffer a, ShortBuffer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,short>") void castunsignedchar2short(@Cast("unsigned char*") byte[] a, short[] b, @Cast("size_t") long size);

public static native @Name("cast<unsigned char,int>") void castunsignedchar2int(@Cast("unsigned char*") BytePointer a, IntPointer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,int>") void castunsignedchar2int(@Cast("unsigned char*") ByteBuffer a, IntBuffer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,int>") void castunsignedchar2int(@Cast("unsigned char*") byte[] a, int[] b, @Cast("size_t") long size);

public static native @Name("cast<unsigned char,long long>") void castunsignedchar2longlong(@Cast("unsigned char*") BytePointer a, LongPointer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,long long>") void castunsignedchar2longlong(@Cast("unsigned char*") ByteBuffer a, LongBuffer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,long long>") void castunsignedchar2longlong(@Cast("unsigned char*") byte[] a, long[] b, @Cast("size_t") long size);

public static native @Name("cast<unsigned char,float>") void castunsignedchar2float(@Cast("unsigned char*") BytePointer a, FloatPointer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,float>") void castunsignedchar2float(@Cast("unsigned char*") ByteBuffer a, FloatBuffer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,float>") void castunsignedchar2float(@Cast("unsigned char*") byte[] a, float[] b, @Cast("size_t") long size);

public static native @Name("cast<unsigned char,double>") void castunsignedchar2double(@Cast("unsigned char*") BytePointer a, DoublePointer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,double>") void castunsignedchar2double(@Cast("unsigned char*") ByteBuffer a, DoubleBuffer b, @Cast("size_t") long size);
public static native @Name("cast<unsigned char,double>") void castunsignedchar2double(@Cast("unsigned char*") byte[] a, double[] b, @Cast("size_t") long size);

public static native @Name("cast<char,unsigned char>") void castchar2unsignedchar(@Cast("char*") BytePointer a, @Cast("unsigned char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<char,unsigned char>") void castchar2unsignedchar(@Cast("char*") ByteBuffer a, @Cast("unsigned char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<char,unsigned char>") void castchar2unsignedchar(@Cast("char*") byte[] a, @Cast("unsigned char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<char,char>") void castchar2char(@Cast("char*") BytePointer a, @Cast("char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<char,char>") void castchar2char(@Cast("char*") ByteBuffer a, @Cast("char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<char,char>") void castchar2char(@Cast("char*") byte[] a, @Cast("char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<char,short>") void castchar2short(@Cast("char*") BytePointer a, ShortPointer b, @Cast("size_t") long size);
public static native @Name("cast<char,short>") void castchar2short(@Cast("char*") ByteBuffer a, ShortBuffer b, @Cast("size_t") long size);
public static native @Name("cast<char,short>") void castchar2short(@Cast("char*") byte[] a, short[] b, @Cast("size_t") long size);

public static native @Name("cast<char,int>") void castchar2int(@Cast("char*") BytePointer a, IntPointer b, @Cast("size_t") long size);
public static native @Name("cast<char,int>") void castchar2int(@Cast("char*") ByteBuffer a, IntBuffer b, @Cast("size_t") long size);
public static native @Name("cast<char,int>") void castchar2int(@Cast("char*") byte[] a, int[] b, @Cast("size_t") long size);

public static native @Name("cast<char,long long>") void castchar2longlong(@Cast("char*") BytePointer a, LongPointer b, @Cast("size_t") long size);
public static native @Name("cast<char,long long>") void castchar2longlong(@Cast("char*") ByteBuffer a, LongBuffer b, @Cast("size_t") long size);
public static native @Name("cast<char,long long>") void castchar2longlong(@Cast("char*") byte[] a, long[] b, @Cast("size_t") long size);

public static native @Name("cast<char,float>") void castchar2float(@Cast("char*") BytePointer a, FloatPointer b, @Cast("size_t") long size);
public static native @Name("cast<char,float>") void castchar2float(@Cast("char*") ByteBuffer a, FloatBuffer b, @Cast("size_t") long size);
public static native @Name("cast<char,float>") void castchar2float(@Cast("char*") byte[] a, float[] b, @Cast("size_t") long size);

public static native @Name("cast<char,double>") void castchar2double(@Cast("char*") BytePointer a, DoublePointer b, @Cast("size_t") long size);
public static native @Name("cast<char,double>") void castchar2double(@Cast("char*") ByteBuffer a, DoubleBuffer b, @Cast("size_t") long size);
public static native @Name("cast<char,double>") void castchar2double(@Cast("char*") byte[] a, double[] b, @Cast("size_t") long size);

public static native @Name("cast<short,unsigned char>") void castshort2unsignedchar(ShortPointer a, @Cast("unsigned char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<short,unsigned char>") void castshort2unsignedchar(ShortBuffer a, @Cast("unsigned char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<short,unsigned char>") void castshort2unsignedchar(short[] a, @Cast("unsigned char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<short,char>") void castshort2char(ShortPointer a, @Cast("char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<short,char>") void castshort2char(ShortBuffer a, @Cast("char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<short,char>") void castshort2char(short[] a, @Cast("char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<short,short>") void castshort2short(ShortPointer a, ShortPointer b, @Cast("size_t") long size);
public static native @Name("cast<short,short>") void castshort2short(ShortBuffer a, ShortBuffer b, @Cast("size_t") long size);
public static native @Name("cast<short,short>") void castshort2short(short[] a, short[] b, @Cast("size_t") long size);

public static native @Name("cast<short,int>") void castshort2int(ShortPointer a, IntPointer b, @Cast("size_t") long size);
public static native @Name("cast<short,int>") void castshort2int(ShortBuffer a, IntBuffer b, @Cast("size_t") long size);
public static native @Name("cast<short,int>") void castshort2int(short[] a, int[] b, @Cast("size_t") long size);

public static native @Name("cast<short,long long>") void castshort2longlong(ShortPointer a, LongPointer b, @Cast("size_t") long size);
public static native @Name("cast<short,long long>") void castshort2longlong(ShortBuffer a, LongBuffer b, @Cast("size_t") long size);
public static native @Name("cast<short,long long>") void castshort2longlong(short[] a, long[] b, @Cast("size_t") long size);

public static native @Name("cast<short,float>") void castshort2float(ShortPointer a, FloatPointer b, @Cast("size_t") long size);
public static native @Name("cast<short,float>") void castshort2float(ShortBuffer a, FloatBuffer b, @Cast("size_t") long size);
public static native @Name("cast<short,float>") void castshort2float(short[] a, float[] b, @Cast("size_t") long size);

public static native @Name("cast<short,double>") void castshort2double(ShortPointer a, DoublePointer b, @Cast("size_t") long size);
public static native @Name("cast<short,double>") void castshort2double(ShortBuffer a, DoubleBuffer b, @Cast("size_t") long size);
public static native @Name("cast<short,double>") void castshort2double(short[] a, double[] b, @Cast("size_t") long size);

public static native @Name("cast<int,unsigned char>") void castint2unsignedchar(IntPointer a, @Cast("unsigned char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<int,unsigned char>") void castint2unsignedchar(IntBuffer a, @Cast("unsigned char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<int,unsigned char>") void castint2unsignedchar(int[] a, @Cast("unsigned char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<int,char>") void castint2char(IntPointer a, @Cast("char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<int,char>") void castint2char(IntBuffer a, @Cast("char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<int,char>") void castint2char(int[] a, @Cast("char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<int,short>") void castint2short(IntPointer a, ShortPointer b, @Cast("size_t") long size);
public static native @Name("cast<int,short>") void castint2short(IntBuffer a, ShortBuffer b, @Cast("size_t") long size);
public static native @Name("cast<int,short>") void castint2short(int[] a, short[] b, @Cast("size_t") long size);

public static native @Name("cast<int,int>") void castint2int(IntPointer a, IntPointer b, @Cast("size_t") long size);
public static native @Name("cast<int,int>") void castint2int(IntBuffer a, IntBuffer b, @Cast("size_t") long size);
public static native @Name("cast<int,int>") void castint2int(int[] a, int[] b, @Cast("size_t") long size);

public static native @Name("cast<int,long long>") void castint2longlong(IntPointer a, LongPointer b, @Cast("size_t") long size);
public static native @Name("cast<int,long long>") void castint2longlong(IntBuffer a, LongBuffer b, @Cast("size_t") long size);
public static native @Name("cast<int,long long>") void castint2longlong(int[] a, long[] b, @Cast("size_t") long size);

public static native @Name("cast<int,float>") void castint2float(IntPointer a, FloatPointer b, @Cast("size_t") long size);
public static native @Name("cast<int,float>") void castint2float(IntBuffer a, FloatBuffer b, @Cast("size_t") long size);
public static native @Name("cast<int,float>") void castint2float(int[] a, float[] b, @Cast("size_t") long size);

public static native @Name("cast<int,double>") void castint2double(IntPointer a, DoublePointer b, @Cast("size_t") long size);
public static native @Name("cast<int,double>") void castint2double(IntBuffer a, DoubleBuffer b, @Cast("size_t") long size);
public static native @Name("cast<int,double>") void castint2double(int[] a, double[] b, @Cast("size_t") long size);

public static native @Name("cast<long long,unsigned char>") void castlonglong2unsignedchar(LongPointer a, @Cast("unsigned char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<long long,unsigned char>") void castlonglong2unsignedchar(LongBuffer a, @Cast("unsigned char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<long long,unsigned char>") void castlonglong2unsignedchar(long[] a, @Cast("unsigned char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<long long,char>") void castlonglong2char(LongPointer a, @Cast("char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<long long,char>") void castlonglong2char(LongBuffer a, @Cast("char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<long long,char>") void castlonglong2char(long[] a, @Cast("char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<long long,short>") void castlonglong2short(LongPointer a, ShortPointer b, @Cast("size_t") long size);
public static native @Name("cast<long long,short>") void castlonglong2short(LongBuffer a, ShortBuffer b, @Cast("size_t") long size);
public static native @Name("cast<long long,short>") void castlonglong2short(long[] a, short[] b, @Cast("size_t") long size);

public static native @Name("cast<long long,int>") void castlonglong2int(LongPointer a, IntPointer b, @Cast("size_t") long size);
public static native @Name("cast<long long,int>") void castlonglong2int(LongBuffer a, IntBuffer b, @Cast("size_t") long size);
public static native @Name("cast<long long,int>") void castlonglong2int(long[] a, int[] b, @Cast("size_t") long size);

public static native @Name("cast<long long,long long>") void castlonglong2longlong(LongPointer a, LongPointer b, @Cast("size_t") long size);
public static native @Name("cast<long long,long long>") void castlonglong2longlong(LongBuffer a, LongBuffer b, @Cast("size_t") long size);
public static native @Name("cast<long long,long long>") void castlonglong2longlong(long[] a, long[] b, @Cast("size_t") long size);

public static native @Name("cast<long long,float>") void castlonglong2float(LongPointer a, FloatPointer b, @Cast("size_t") long size);
public static native @Name("cast<long long,float>") void castlonglong2float(LongBuffer a, FloatBuffer b, @Cast("size_t") long size);
public static native @Name("cast<long long,float>") void castlonglong2float(long[] a, float[] b, @Cast("size_t") long size);

public static native @Name("cast<long long,double>") void castlonglong2double(LongPointer a, DoublePointer b, @Cast("size_t") long size);
public static native @Name("cast<long long,double>") void castlonglong2double(LongBuffer a, DoubleBuffer b, @Cast("size_t") long size);
public static native @Name("cast<long long,double>") void castlonglong2double(long[] a, double[] b, @Cast("size_t") long size);

public static native @Name("cast<float,unsigned char>") void castfloat2unsignedchar(FloatPointer a, @Cast("unsigned char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<float,unsigned char>") void castfloat2unsignedchar(FloatBuffer a, @Cast("unsigned char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<float,unsigned char>") void castfloat2unsignedchar(float[] a, @Cast("unsigned char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<float,char>") void castfloat2char(FloatPointer a, @Cast("char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<float,char>") void castfloat2char(FloatBuffer a, @Cast("char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<float,char>") void castfloat2char(float[] a, @Cast("char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<float,short>") void castfloat2short(FloatPointer a, ShortPointer b, @Cast("size_t") long size);
public static native @Name("cast<float,short>") void castfloat2short(FloatBuffer a, ShortBuffer b, @Cast("size_t") long size);
public static native @Name("cast<float,short>") void castfloat2short(float[] a, short[] b, @Cast("size_t") long size);

public static native @Name("cast<float,int>") void castfloat2int(FloatPointer a, IntPointer b, @Cast("size_t") long size);
public static native @Name("cast<float,int>") void castfloat2int(FloatBuffer a, IntBuffer b, @Cast("size_t") long size);
public static native @Name("cast<float,int>") void castfloat2int(float[] a, int[] b, @Cast("size_t") long size);

public static native @Name("cast<float,long long>") void castfloat2longlong(FloatPointer a, LongPointer b, @Cast("size_t") long size);
public static native @Name("cast<float,long long>") void castfloat2longlong(FloatBuffer a, LongBuffer b, @Cast("size_t") long size);
public static native @Name("cast<float,long long>") void castfloat2longlong(float[] a, long[] b, @Cast("size_t") long size);

public static native @Name("cast<float,float>") void castfloat2float(FloatPointer a, FloatPointer b, @Cast("size_t") long size);
public static native @Name("cast<float,float>") void castfloat2float(FloatBuffer a, FloatBuffer b, @Cast("size_t") long size);
public static native @Name("cast<float,float>") void castfloat2float(float[] a, float[] b, @Cast("size_t") long size);

public static native @Name("cast<float,double>") void castfloat2double(FloatPointer a, DoublePointer b, @Cast("size_t") long size);
public static native @Name("cast<float,double>") void castfloat2double(FloatBuffer a, DoubleBuffer b, @Cast("size_t") long size);
public static native @Name("cast<float,double>") void castfloat2double(float[] a, double[] b, @Cast("size_t") long size);

public static native @Name("cast<double,unsigned char>") void castdouble2unsignedchar(DoublePointer a, @Cast("unsigned char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<double,unsigned char>") void castdouble2unsignedchar(DoubleBuffer a, @Cast("unsigned char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<double,unsigned char>") void castdouble2unsignedchar(double[] a, @Cast("unsigned char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<double,char>") void castdouble2char(DoublePointer a, @Cast("char*") BytePointer b, @Cast("size_t") long size);
public static native @Name("cast<double,char>") void castdouble2char(DoubleBuffer a, @Cast("char*") ByteBuffer b, @Cast("size_t") long size);
public static native @Name("cast<double,char>") void castdouble2char(double[] a, @Cast("char*") byte[] b, @Cast("size_t") long size);

public static native @Name("cast<double,short>") void castdouble2short(DoublePointer a, ShortPointer b, @Cast("size_t") long size);
public static native @Name("cast<double,short>") void castdouble2short(DoubleBuffer a, ShortBuffer b, @Cast("size_t") long size);
public static native @Name("cast<double,short>") void castdouble2short(double[] a, short[] b, @Cast("size_t") long size);

public static native @Name("cast<double,int>") void castdouble2int(DoublePointer a, IntPointer b, @Cast("size_t") long size);
public static native @Name("cast<double,int>") void castdouble2int(DoubleBuffer a, IntBuffer b, @Cast("size_t") long size);
public static native @Name("cast<double,int>") void castdouble2int(double[] a, int[] b, @Cast("size_t") long size);

public static native @Name("cast<double,long long>") void castdouble2longlong(DoublePointer a, LongPointer b, @Cast("size_t") long size);
public static native @Name("cast<double,long long>") void castdouble2longlong(DoubleBuffer a, LongBuffer b, @Cast("size_t") long size);
public static native @Name("cast<double,long long>") void castdouble2longlong(double[] a, long[] b, @Cast("size_t") long size);

public static native @Name("cast<double,float>") void castdouble2float(DoublePointer a, FloatPointer b, @Cast("size_t") long size);
public static native @Name("cast<double,float>") void castdouble2float(DoubleBuffer a, FloatBuffer b, @Cast("size_t") long size);
public static native @Name("cast<double,float>") void castdouble2float(double[] a, float[] b, @Cast("size_t") long size);

public static native @Name("cast<double,double>") void castdouble2double(DoublePointer a, DoublePointer b, @Cast("size_t") long size);
public static native @Name("cast<double,double>") void castdouble2double(DoubleBuffer a, DoubleBuffer b, @Cast("size_t") long size);
public static native @Name("cast<double,double>") void castdouble2double(double[] a, double[] b, @Cast("size_t") long size);

// #endif

}
