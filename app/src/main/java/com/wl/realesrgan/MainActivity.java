package com.wl.realesrgan;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity implements View.OnClickListener {
    private static final String TAG = "MainActivity";
    private DetectionManager mDetectionManager;
    private Bitmap mBitmap;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mDetectionManager = new DetectionManager(this);
        ImageView inputImage = findViewById(R.id.imageView1);
        mBitmap = BitmapFactory.decodeStream(readInputImage());
        inputImage.setImageBitmap(mBitmap);
        findViewById(R.id.super_resolution_button).setOnClickListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mDetectionManager.onDestroy();
    }

    private InputStream readInputImage() {
        try {
            return getAssets().open("test_superresolution.png");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onClick(View v) {
        final int id = v.getId();
        if (id == R.id.super_resolution_button) {
            try {
                long start = System.currentTimeMillis();
                performSuperResolution();
                long end = System.currentTimeMillis();
                long cost = end - start;
                Toast.makeText(this, "cost = " + cost + "ms", Toast.LENGTH_SHORT)
                        .show();
            } catch (Exception e) {
                Log.e(TAG, "Exception caught when perform super resolution", e);
                Toast.makeText(this, "Failed to perform super resolution", Toast.LENGTH_SHORT)
                        .show();
            }
        }
    }

    private void performSuperResolution() {
        updateUI(mDetectionManager.run(mBitmap, 512));
    }

    private void updateUI(Bitmap bitmap) {
        ImageView outputImage = findViewById(R.id.imageView2);
        outputImage.setImageBitmap(bitmap);
    }
}