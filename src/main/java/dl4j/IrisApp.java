package dl4j;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class IrisApp {

	public static void main(String[] args) throws Exception {
		// Configuration du modele MultiLayerPerceptron
		// le seed correspondra a la valeur aleatoire qui sera utilise pour representer
		// le poids de chaque neuronne
		// Adam permet de corriger l'erreur de sortie, cad, l'erreur resultant de la
		// sortie reelle - sortie attendue
		// Puis nous ajoutons les couches
		// couche 0 (layer(0)) est dite fully connector (DenseLayer), car la sortie de
		// chaque neuronne de la couche deviens l'entree de tous les neuronne de la
		// couche suivante.

		double learningRate = 0.001;
		int inputSize = 4;
		int numHiddenNodes = 10;
		int outputSize = 3;
		int batchSize = 1;
		int classIndex = 4;
		int numEpochs = 60;
		String [] labels = {"Iris-setosa", "Iris-versicolor","Iris-virginica"};

		System.out.println("Creation du modele");
		MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(1234)
				.updater(new Adam(learningRate)).list()
				.layer(0,
						new DenseLayer.Builder().nIn(inputSize).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())
				.layer(1,
						new OutputLayer.Builder().nIn(numHiddenNodes).nOut(outputSize)
								.lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
								.activation(Activation.SOFTMAX).build())
				.build();

		// Creation du modele
		MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
		multiLayerNetwork.init();
		// System.out.println(multiLayerConfiguration.toJson());

		// Demarrage du serveur de monitoring du processus d'apprentissage avec UIServer
		// utilise le port 9000 par defaut
		UIServer uiServer = UIServer.getInstance();
		InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
		uiServer.attach(inMemoryStatsStorage);

		multiLayerNetwork.setListeners(new StatsListener(inMemoryStatsStorage));

		// Lecture du fichier CSV et Entrainement du modele
		System.out.println("Entrainement du modele");
		File fileTrain = new ClassPathResource("iris-train.csv").getFile();
		RecordReader recordReaderTrain = new CSVRecordReader();
		recordReaderTrain.initialize(new FileSplit(fileTrain));
		// organisation dans un dataset Iterator

		DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex,
				outputSize);
		// Parcour du dataset batch par batch

		/*
		 * while(dataSetIteratorTrain.hasNext()) { System.out.println(
		 * "------------------------------------------------------------"); DataSet
		 * dataSet = dataSetIteratorTrain.next();
		 * System.out.println(dataSet.getFeatures());
		 * System.out.println(dataSet.getLabels()); }
		 */

		for (int i = 0; i < numEpochs; i++) {
			multiLayerNetwork.fit(dataSetIteratorTrain);
		}

		// Evaluation du modele (avec iris-test.csv)
		System.out.println("Evaluation du modele");
		File fileTest = new ClassPathResource("irisTest.csv").getFile();
		RecordReader recordReaderTest = new CSVRecordReader();
		recordReaderTest.initialize(new FileSplit(fileTest));
		DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, classIndex,
				outputSize);

		Evaluation evaluation = new Evaluation();

		while (dataSetIteratorTest.hasNext()) {
			DataSet dataSetTest = dataSetIteratorTest.next();
			INDArray features = dataSetTest.getFeatures();
			INDArray targetLabels = dataSetTest.getLabels();
			INDArray predictedLabels = multiLayerNetwork.output(features);
			evaluation.eval(predictedLabels, targetLabels);
		}
		System.out.println(evaluation.stats());
		System.out.println("Prédictions");

		INDArray inputData = Nd4j.create(new double[][] { { 5.1, 3.5, 1.4, 0.2 }, { 4.9, 3.0, 1.4, 0.2 },
				{ 6.7, 3.1, 4.4, 1.4 }, { 5.6, 3.0, 4.5, 1.5 }, { 6.0, 3.0, 4.5, 1.8 }, { 6.9, 3.1, 5.4, 2.1 } });
		INDArray output = multiLayerNetwork.output(inputData);
		
		int [] classesIris = output.argMax(1).toIntVector();
		for (int i = 0; i < classesIris.length; i++) {
			System.out.println("Classe d'iris :" + labels[classesIris[i]]);
		}
		
		// System.out.println(output);
		
		
		

	}
}
