import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Operation;


public class OnnxLoader {
    public static void main(String[] args) {
        // put the actual directory and filename of your model here
        byte[] graphDef = readAllBytesOrExit(Paths.get("/home/stefan/Hiwi/NeuralIntegration/onnx_to_tensorflow", "test.pb"));

        // put the actual shape of your input placeholder(s) here,
        // and initialize with your actual input data
        //float[][] input = new float[64][1000];
        float[][][] inputs = {{{-0.5525f, 0.6355f, -0.3968f}},{{-0.6571f, -1.6428f, 0.9803f}},{{-0.0421f, -0.8206f, 0.3133f}},{{-1.1352f, 0.3773f, -0.2824f}},{{-2.5667f, -1.4303f, 0.5009f}}};
        float[][][] initial_h = {{{0.5438f, -0.4057f,  1.1341f}}};
        float[][][] initial_c = {{{-1.1115f, 0.3501f, -0.7703f}}};

        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);

            try (Session s = new Session(g)) {
                Tensor tInputs = Tensor.create(inputs);
                Tensor tInitialH = Tensor.create(initial_h);
                Tensor tInitialC = Tensor.create(initial_c);

                // feed your actual input placeholders, and fetch
                // your actual output tensors here, using the
                // actual dtype of your output data
                Tensor<Float> tResult = s.runner().feed("0:0", tInputs).feed("1:0", tInitialH).feed("2:0", tInitialC).fetch("Squeeze_3:0").run().get(0).expect(Float.class);
		
                // convert as required by your actual
                // output tensor(s)
                float[][][] result = new float[5][1][3];
				tResult.copyTo(result);
		
                for( int i = 0; i < result.length; i++ ) {
                    for( int j = 0; j < result[i].length; j++ ) {
                        String outp = "";
                        for( int k = 0; k < result[i][j].length; k++ ) {
                            outp += Float.toString(result[i][j][k]) + " ";
                        }
                        System.out.println(outp);
                    }
                }
            }
        }
    }

    // from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }
}
