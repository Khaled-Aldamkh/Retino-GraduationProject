package com.example.retino;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;

public class ImageUtils {
    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        Bitmap finalBitmap = Bitmap.createScaledBitmap(
                bitmap,
               224,
               224,
                false);
        return finalBitmap;
    }
}
