package com.wl.realesrgan;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class DetectionManager {
    private static final String TAG = "UpSmpler";
    private static final int PILE_PAD = 10;
    private static final int SCALE = 4;

    private final Resources mResources;
    private final OrtEnvironment mOrtEnvironment;
    private OrtSession mOrtSession;
    private final String mInputName;

    public DetectionManager(Context context) {
        mResources = context.getResources();
        mOrtEnvironment = OrtEnvironment.getEnvironment();
        mOrtSession = createOrtSession();
        mInputName = mOrtSession.getInputNames().iterator().next();
    }

    public void onDestroy() {
        try {
            mOrtSession.close();
            mOrtSession = null;
        } catch (Exception e) {
            Log.e(TAG, "WL_DEBUG onDestroy e = " + e, e);
        }
        mOrtEnvironment.close();
    }

    public Bitmap run(Bitmap image, int tileSize) {
        int height = image.getHeight();
        int width = image.getWidth();
        final FloatBuffer srcFloatBuffer = Utils.bitmapToFloatBuffer(image);
        OnnxTensor tensor = createTensor(srcFloatBuffer, new long[]{1, 3, height, width});
        final int[] pixels = tileProcess(tensor, tileSize);
        int outputHeight = height * SCALE;
        int outputWidth = width * SCALE;
        Bitmap img = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
        img.setPixels(pixels, 0, outputWidth, 0, 0, outputWidth, outputHeight);
        return img;
    }

    private int[] tileProcess(OnnxTensor tensor, int tileSize) {
        long[] shape = tensor.getInfo().getShape();
        int height = (int) shape[2];
        int width = (int) shape[3];
        int outputHeight = height * SCALE;
        int outputWidth = width * SCALE;
        final int pixelsCount = outputHeight * outputWidth;
        final int[] pixels = new int[pixelsCount];
        int tiles_x = (int) Math.ceil((float) width / tileSize);
        int tiles_y = (int) Math.ceil((float) height / tileSize);
        Log.i(TAG, "WL_DEBUG tileProcess tiles_x = " + tiles_x + ", tiles_y = " + tiles_y);
        for (int y = 0; y < tiles_y; y++) {
            for (int x = 0; x < tiles_x; x++) {
                int ofs_x = x * tileSize;
                int ofs_y = y * tileSize;
                int input_end_x = Math.min(ofs_x + tileSize, width);
                int input_end_y = Math.min(ofs_y + tileSize, height);
                int input_start_x_pad = Math.max(ofs_x - PILE_PAD, 0);
                int input_end_x_pad = Math.min(input_end_x + PILE_PAD, width);
                int input_start_y_pad = Math.max(ofs_y - PILE_PAD, 0);
                int input_end_y_pad = Math.min(input_end_y + PILE_PAD, height);
                OnnxTensor input_tile = readTile(tensor, input_start_y_pad, input_end_y_pad, input_start_x_pad, input_end_x_pad);
                OrtSession.Result output = null;
                try {
                    output = mOrtSession.run(Collections.singletonMap(mInputName, input_tile));
                } catch (OrtException e) {
                    Log.e(TAG, "WL_DEBUG tileProcess e = " + e, e);
                }
                float[][][][] output_tile = null;
                try {
                    output_tile = (float[][][][]) output.get(0).getValue();
                } catch (OrtException e) {
                    Log.e(TAG, "WL_DEBUG tileProcess e = " + e, e);
                }

                int output_start_x = ofs_x * SCALE;
                int output_end_x = input_end_x * SCALE;
                int output_start_y = ofs_y * SCALE;
                int output_end_y = input_end_y * SCALE;
                int output_start_x_tile = (ofs_x - input_start_x_pad) * SCALE;
                int output_start_y_tile = (ofs_y - input_start_y_pad) * SCALE;
                writeTile(pixels, output_start_y, output_end_y, output_start_x, output_end_x, output_tile, output_start_y_tile, output_start_x_tile, outputWidth);
                Log.i(TAG, "WL_DEBUG tileProcess x = " + x + ", y = " + y);
            }
        }

        return pixels;
    }

    private OnnxTensor readTile(OnnxTensor input, int start1, int end1, int start2, int end2) {
        final int height = end1 - start1;
        final int width = end2 - start2;
        final long[] inputShape = input.getInfo().getShape();
        final int inputHeight = (int) inputShape[2];
        final int inputWidth = (int) inputShape[3];
        final int inputPixelsCount = inputHeight * inputWidth;
        final int inputOffset_b = 2 * inputPixelsCount;
        final FloatBuffer inputFloatBuffer = input.getFloatBuffer();
        final FloatBuffer outputFloatBuffer = Utils.allocateFloatBuffer(3 * height * width);
        final int pixelsCount = height * width;
        final int offset_b = 2 * pixelsCount;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = width * i + j;
                int inputIndex = (i + start1) * inputWidth + (j + start2);
                outputFloatBuffer.put(index, inputFloatBuffer.get(inputIndex));
                outputFloatBuffer.put(pixelsCount + index, inputFloatBuffer.get(inputPixelsCount + inputIndex));
                outputFloatBuffer.put(offset_b + index, inputFloatBuffer.get(inputOffset_b + inputIndex));
            }
        }
        return createTensor(outputFloatBuffer, new long[]{1, 3, height, width});
    }

    private void writeTile(int[] dest, int start1, int end1, int start2, int end2, float[][][][] input, int start3, int start4, int width) {
        int destHeight = end1 - start1;
        int destWidth = end2 - start2;
        for (int i = 0; i < destHeight; i++) {
            for (int j = 0; j < destWidth; j++) {
                int index = width * (i + start1) + (j + start2);
                int red = (int) Math.max(Math.min(input[0][0][i + start3][j + start4] * 255, 255), 0);
                int green = (int) Math.max(Math.min(input[0][1][i + start3][j + start4] * 255, 255), 0);
                int blue = (int) Math.max(Math.min(input[0][2][i + start3][j + start4] * 255, 255), 0);
                dest[index] = (0xFF << 24) | (red << 16) | (green << 8) | (blue);
            }
        }
    }

    private OrtSession createOrtSession() {
        OrtSession result = null;
        try {
            OrtSession.SessionOptions option = new OrtSession.SessionOptions();
            result = mOrtEnvironment.createSession(readModel(), option);
        } catch (Exception e) {
            Log.e(TAG, "WL_DEBUG createOrtSession e = " + e, e);
        }
        return result;
    }

    private byte[] readModel() {
        int id = R.raw.realesr_general_x4v3;//R.raw.realesrgan_x4plus;
        InputStream is = mResources.openRawResource(id);
        byte[] result = null;
        try {
            result = Utils.readAllBytes(is);
        } catch (Exception e) {
            Log.e(TAG, "WL_DEBUG readModel e = " + e, e);
        }
        try {
            is.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return result;
    }



    private OnnxTensor createTensor(FloatBuffer data, long[] shape) {
        OnnxTensor result = null;
        try {
            result = OnnxTensor.createTensor(mOrtEnvironment, data, shape);
        } catch (OrtException e) {
            Log.e(TAG, "WL_DEBUG run forward e = " + e, e);
        }
        return result;
    }
}
