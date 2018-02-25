import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;


public class OnnxLoader {
    public static void main(String[] args) {
	// put the actual directory and filename of your model here
        byte[] graphDef = readAllBytesOrExit(Paths.get("/tmp/my-model", "train.pb"));

	// put the actual shape of your input placeholder(s) here,
	// and initialize with your actual input data
        float[][] input = new float[64][1000];

        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
	    
            try (Session s = new Session(g)) {
                Tensor tInput = Tensor.create(input);

		// feed your actual input placeholders, and fetch
		// your actual output tensors here, using tye
		// actual dtype of your output data
                Tensor<Float> tResult = s.runner().feed("input_placeholder", tInput).fetch("output_tensor").run().get(0).expect(Float.class);
		
                // convert as required by your actual
		// output tensor(s)
                float[][] result = new float[64][10];
                tResult.copyTo(result);
		
                for( int i = 0; i < result.length; i++ ) {
                    System.err.println(Arrays.toString(result[i]));
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
