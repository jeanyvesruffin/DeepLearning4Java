package dl4j;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class IrisPrediction {

	public static void main(String[] args) throws IOException {
		String[] labels = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
		System.out.println("Chargement du modele");
		MultiLayerNetwork modelToLoad = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));
		System.out.println("Prédictions");
		INDArray inputData = Nd4j.create(new double[][] { { 5.1, 3.5, 1.4, 0.2 }, { 4.9, 3.0, 1.4, 0.2 },
				{ 6.7, 3.1, 4.4, 1.4 }, { 5.6, 3.0, 4.5, 1.5 }, { 6.0, 3.0, 4.5, 1.8 }, { 6.9, 3.1, 5.4, 2.1 } });
		INDArray output = modelToLoad.output(inputData);
		int[] classesIris = output.argMax(1).toIntVector();
		for (int i = 0; i < classesIris.length; i++) {
			System.out.println("Classe d'iris :" + labels[classesIris[i]]);
		}

	}

}
