package com.wl.realesrgan;

import android.graphics.Bitmap;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {
	private static final int DEFAULT_BUFFER_SIZE = 8192;
	private static final int MAX_BUFFER_SIZE = Integer.MAX_VALUE - 8;
	private static final int FLOAT_SIZE_BYTES = 4;

	public static byte[] readAllBytes(InputStream is) throws IOException {
		return readNBytes(is, Integer.MAX_VALUE);
	}

	private static byte[] readNBytes(InputStream is, int len) throws IOException {
		if (len < 0) {
			throw new IllegalArgumentException("len < 0");
		}

		List<byte[]> bufs = null;
		byte[] result = null;
		int total = 0;
		int remaining = len;
		int n;
		do {
			byte[] buf = new byte[Math.min(remaining, DEFAULT_BUFFER_SIZE)];
			int nread = 0;

			// read to EOF which may read more or less than buffer size
			while ((n = is.read(buf, nread,
					Math.min(buf.length - nread, remaining))) > 0) {
				nread += n;
				remaining -= n;
			}

			if (nread > 0) {
				if (MAX_BUFFER_SIZE - total < nread) {
					throw new OutOfMemoryError("Required array size too large");
				}
				if (nread < buf.length) {
					buf = Arrays.copyOfRange(buf, 0, nread);
				}
				total += nread;
				if (result == null) {
					result = buf;
				} else {
					if (bufs == null) {
						bufs = new ArrayList<>();
						bufs.add(result);
					}
					bufs.add(buf);
				}
			}
			// if the last call to read returned -1 or the number of bytes
			// requested have been read then break
		} while (n >= 0 && remaining > 0);

		if (bufs == null) {
			if (result == null) {
				return new byte[0];
			}
			return result.length == total ?
					result : Arrays.copyOf(result, total);
		}

		result = new byte[total];
		int offset = 0;
		remaining = total;
		for (byte[] b : bufs) {
			int count = Math.min(b.length, remaining);
			System.arraycopy(b, 0, result, offset, count);
			offset += count;
			remaining -= count;
		}

		return result;
	}

	public static FloatBuffer bitmapToFloatBuffer(
			final Bitmap bitmap) {
		int x = 0;
		int y = 0;
		int width = bitmap.getWidth();
		int height = bitmap.getHeight();

		final FloatBuffer floatBuffer = allocateFloatBuffer(3 * width * height);
		bitmapToFloatBuffer(
				bitmap, x, y, width, height, floatBuffer);
		return floatBuffer;
	}

	public static FloatBuffer allocateFloatBuffer(int numElements) {
		return ByteBuffer.allocateDirect(numElements * FLOAT_SIZE_BYTES)
				.order(ByteOrder.nativeOrder())
				.asFloatBuffer();
	}

	private static void bitmapToFloatBuffer(
			final Bitmap bitmap,
			final int x,
			final int y,
			final int width,
			final int height,
			final FloatBuffer outBuffer) {
		final int pixelsCount = height * width;
		final int[] pixels = new int[pixelsCount];
		bitmap.getPixels(pixels, 0, width, x, y, width, height);
		final int offset_b = 2 * pixelsCount;
		for (int i = 0; i < pixelsCount; i++) {
			final int c = pixels[i];
			float r = ((c >> 16) & 0xff) / 255.0f;
			float g = ((c >> 8) & 0xff) / 255.0f;
			float b = ((c) & 0xff) / 255.0f;
			outBuffer.put(i, r);
			outBuffer.put(pixelsCount + i, g);
			outBuffer.put(offset_b + i, b);
		}
	}
}
