package com.example.retino;
// import libraries
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static java.lang.Math.min;

class Classification {
    public final String title;
    public final float confidence;

    public Classification(String title, float confidence) {
        this.title = title;
        this.confidence = confidence;
    }

    @Override
    public String toString() {
        return title + " " + String.format("(%.1f%%) ", confidence * 100.0f);
    }
}

public class MainActivity extends AppCompatActivity {

    protected Interpreter tflite;
    private int imgWidth=224;
    private int imgHeight=224;
    private static final int MAX_RESULTS = 3;
    public static final float CLASSIFICATION_THRESHOLD = 0.2f;
    private static final float IMAGE_MEAN = 0;
    private static final float IMAGE_STD = 255.0f;
    private Bitmap bitmap;
    ImageView imageView;
    private Uri imageuri;
    Button buclassify;
    TextView classitext;
    private static final String TAG = MainActivity.class.getName();

    // Class labels
    public static final List<String> OUTPUT_LABELS = Collections.unmodifiableList(
            Arrays.asList(
                    "0 - No DR",
                    "1 - Mild DR",
                    "2 - Moderate DR",
                    "3 - Severe DR",
                    "4 - Proliferative DR"
            ));

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView=(ImageView)findViewById(R.id.image);
        buclassify=(Button)findViewById(R.id.classify);
        classitext=(TextView)findViewById(R.id.classifytext);
        findViewById(R.id.classify).setEnabled(false);


        try{
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(5);
            options.setUseNNAPI(true);

            tflite=new Interpreter(loadmodelfile(this),options);

        }catch (Exception e) {
            Toast.makeText(this,e.toString(),Toast.LENGTH_LONG).show();
        }

        buclassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Bitmap preprocessedImage = ImageUtils.prepareImageForClassification(bitmap);
                List<Classification> recognitions = recognizeImage(preprocessedImage);
                classitext.setText(recognitions.toString());


            }
        });



    }

// using A mapped byte buffer and the file mapping that it represents remain valid until the buffer itself is garbage-collected.
    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor=activity.getAssets().openFd("newmodel.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    public List<Classification> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float[][] result = new float[1][OUTPUT_LABELS.size()];
        tflite.run(byteBuffer, result);
        return getSortedResult(result);
    }

// For reading, need to convert
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imgWidth*imgHeight*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[imgWidth * imgHeight];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < imgWidth; ++i) {
            for (int j = 0; j < imgHeight; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    private List<Classification> getSortedResult(float[][] resultsArray) {

        PriorityQueue<Classification> sortedResults = new PriorityQueue<>(
                MAX_RESULTS,
                (lhs, rhs) -> Float.compare(rhs.confidence, lhs.confidence)
        );

        for (int i = 0; i < OUTPUT_LABELS.size(); i++) {
            float confidence = resultsArray[0][i];

            if (confidence >= CLASSIFICATION_THRESHOLD) {
                sortedResults.add(new Classification(OUTPUT_LABELS.get(i), confidence));
            }
        }
        return new ArrayList<>(sortedResults);

    }


    public void checkPermission(String permission, int requestCode) {
        if (ContextCompat.checkSelfPermission(MainActivity.this, permission) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[] { permission }, requestCode);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == 101) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(MainActivity.this, "Camera Permission Granted", Toast.LENGTH_SHORT) .show();
            }
            else {
                Toast.makeText(MainActivity.this, "Camera Permission Denied", Toast.LENGTH_SHORT) .show();
            }
        }
    }


    public void getImg(View view) {
        checkPermission(Manifest.permission.CAMERA, 101);
        final CharSequence[] options = {"Capture Photo", "Choose from Gallery"};

        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("Upload Picture");
        builder.setItems(options, new DialogInterface.OnClickListener() {

            @Override
            public void onClick(DialogInterface dialog, int item) {

                if (options[item].equals("Capture Photo")) {
                    Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    // Create the File where the photo should go
                    File photoFile = null;
                    try {
                        photoFile = createTempFile();
                        if(photoFile == null)
                            throw  new IOException();
                    } catch (IOException e) {
                        e.printStackTrace();
                        // Error occurred while creating the File
                    }
                    // Continue only if the File was successfully created
                    if (photoFile != null) {
                        imageuri = FileProvider.getUriForFile(MainActivity.this,
                                "com.example.retino.FileProvider",
                                photoFile);
                        takePicture.putExtra(MediaStore.EXTRA_OUTPUT, imageuri);
                        startActivityForResult(takePicture, 0);
                    }
                }

                else if (options[item].equals("Choose from Gallery")) {
                    Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(pickPhoto , 1);

                }
            }
        });
        builder.show();
    }
    public File createTempFile() throws IOException {

        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        String imageFileName = "/JPEG_PIC.jpg";
        return new File(storageDir.getPath()+imageFileName);
    }


    @Override
    protected void onActivityResult(int reqCode, int resCode, Intent data) {
        super.onActivityResult(reqCode, resCode, data);
        switch (reqCode) {
            case 0:
                if (resCode == RESULT_OK ) {
                    Bitmap bitmap = null;
                    try {
                       bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageuri);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    this.bitmap = bitmap;
                    imageView.setImageBitmap(bitmap);
                    findViewById(R.id.classify).setEnabled(true);

                }
                break;
            case 1:
                if (resCode == RESULT_OK && data != null) {
                    Uri imageUri = data.getData();
                    Bitmap bitmap = null;
                    try {
                        bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    this.bitmap = bitmap;
                    imageView.setImageBitmap(bitmap);
                    findViewById(R.id.classify).setEnabled(true);

                }
        }


    }
}

