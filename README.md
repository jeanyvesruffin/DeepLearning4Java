# Deep Learning For Java (DL4J)

[https://deeplearning4j.konduit.ai/](https://deeplearning4j.konduit.ai/)


## Definition

**Deep Learning 4 java** est un framework open source (licence apache) qui permet de construire, entrainer et tester une grande diversite d'algorithmes de Deep Learning (depuis les reseaux standard, jusqu'aux reseau concolutionels, en passant par des architectures plus complexes).

Il se base sur sa structure de donnees (Nd4j) permettant d'effectuer les operations de l'algebres lineaires sur les architectures massivement paralleles GPU et les architectures distribuees.

**Nd4j** utilise du code natif (Cuda), et alloue de l'espace or du tas Java. Ceci est imperativement a prendre en compte lorsque la volumetrie des donnees est importante.

DL4J utilise **DataVec** pour la vectorisation et la transformation des donnes.

## But de l'application

Le but est de creer un modele d'entrainement supervise a l'aide du dataset Iris. Le dataset fournis 120 instances (120 lignes) d'exemples de donnees de fleurs, classees en 3 types de fleurs (iris-setosa, iris-versicolor et iris virginica) et pour chaque type de fleurs nous avons 4 carateristiques (SepalLength, SepalWidth, PetalLength et PetalWidth)


<img src="src/main/resources/images/Iris-setosa.png"  width="120" height="120"><img src="src/main/resources/images/Iris-versicolor.png"  width="120" height="120"><img src="src/main/resources/images/Iris-virginica.png"  width="120" height="120">

## Description du dataset (iris-train.csv)

* La 1er colonne correspond au sepalLength
* La 2eme au sepalWidth
* La 3eme au petalLength
* La 4eme au petalWidth
* La 5eme correspond au type de fleurs (0 pour setosa, 1 pour versicolor et 2 pour virginica)

## Retour sur le modele

Le modele de reseau de neuronnes que nous allons creer, et un multilayer perceptron.

>Un multilayer perceptron ou perceptron multicouche (multilayer perceptron MLP) est un type de reseau neuronal artificiel organise en plusieurs couches au sein desquelles une information circule de la couche d'entree vers la couche de sortie uniquement ; il s'agit donc d'un reseau a propagation directe (feedforward). Chaque couche est constituee d'un nombre variable de neurones, les neurones de la derniere couche (dite de sortie) etant les sorties du systeme global.


**Modele MLP**

<img src="src/main/resources/images/model.png"  width="400" height="400">

Ce MLP sera compose de 3 couches:

* l'input ou il y a 4 entrees
* une couche masquee, Sigmoid, ou nous utiliserons un certain nombre de neuronnes (10, 20, 30 ...) pour verifier la precision des modeles.
* output, SoftMax, ou il y a 3 sortie, correspondants a la probabilite que la sortie soit un iris setosa, ou versicolor ou virginica.

Nous utiliserons plusieurs parametrage, par exemple pour :

* La couche masquee nous utiliserons une fonction d'activation de sigmoid:

>En mathematiques, la fonction sigmoide (dite aussi courbe en S) represente la fonction de repartition de la loi logistique. Elle est souvent utilisee dans les reseaux de neurones parce qu'elle est derivable, ce qui est une contrainte pour l'algorithme de retropropagation de Werbos. La forme de la derivee de sa fonction inverse est extremement simple et facile a calculer, ce qui ameliore les performances des algorithmes.
La courbe sigmoide genere par transformation affine une partie des courbes logistiques et en est donc un representant privilegie.

* pour la couche de sortie nous utiliserons une fonction d'activation de softMax. Nous permettant de nous donner la probabilite que l'exemple fournis appartiens bien a une classe precise (type d'iris), si l'on fait la sommes des sortie celle-ci sera egale a 1.

>En mathematiques, la fonction softmax, ou fonction exponentielle normalisee, est une generalisation de la fonction logistique qui prend en entree un vecteur.
En theorie des probabilites, la sortie de la fonction softmax peut etre utilisee pour representer une loi categorielle c'est-a-dire une loi de probabilite sur K differents resultats possibles.
La fonction softmax est egalement connue pour etre utilisee dans diverses methodes de classification en classes multiples, par exemple dans le cas de reseaux de neurones artificiels.

* Retropropagation du gradient (fonction MearnSquaredError et Optimiser: ADAM). Le principe est que l'on donne des exemples dont on connait la sortie. La sortie predite moins la sortie reelle nous permet de determiner l'erreur (un delta). Il faut minimiser l'erreur en utilisant la fonction MearnSquaredError, cad, l'erreur quadratique.

>En statistiques, l'erreur quadratique moyenne  d'un estimateur d'un parametre de dimension 1 (mean squared error) est une mesure caracterisant la precision de cet estimateur. Elle est plus souvent appelee erreur quadratique  (moyenne etant sous-entendu) ; elle est parfois appelee aussi  risque quadratique.


>En statistiques, la retropropagation du gradient est une methode pour calculer le gradient de l'erreur pour chaque neurone d'un reseau de neurones, de la derniere couche vers la premiere. De facon abusive, on appelle souvent technique de retropropagation du gradient l'algorithme classique de correction des erreurs base sur le calcul du gradient grace a la retropropagation et c'est cette methode qui est presentee ici. En verite, la correction des erreurs peut se faire selon d'autres methodes, en particulier le calcul de la derivee seconde. Cette technique consiste a corriger les erreurs selon l'importance des elements qui ont justement participe a la realisation de ces erreurs. Dans le cas des reseaux de neurones, les poids synaptiques qui contribuent a engendrer une erreur importante se verront modifies de maniere plus significative que les poids qui ont engendre une erreur marginale.
Ce principe fonde les methodes de type algorithme du gradient, qui sont efficacement utilisees dans des reseaux de neurones multicouches comme les perceptrons multicouches. L'algorithme du gradient a pour but de converger de maniere iterative vers une configuration optimisee des poids synaptiques. Cet etat peut etre un minimum local de la fonction a optimiser et idealement, un minimum global de cette fonction (dite fonction de cout).
Normalement, la fonction de cout est non lineaire au regard des poids synaptiques. Elle dispose egalement d'une borne inferieure et moyennant quelques precautions lors de l'apprentissage, les procedures d'optimisation finissent par aboutir a une configuration stable au sein du reseau de neurones.

* Vitesse d'apprentissage : Learning Rate aura la valeur de 0.001.

## Description de l'application

L'application devra avoir plusieurs onglets (Front en javaFx).

L'application nous permettra de creer un modele (Create Model), charger les donnees (Load data) necessaire pour  entrainer le model, entrainer le modele (Train Model) 80% du dataset, evaluer le modele (Evaluate Model) 20% du dataset, faire les predictions (Predict) et enfin la sauvegarde du model (save Model) et le chargement du model (load Model).

Enfin, un formulaire sera present pour renseigner les observations en input et a l'aide d'un bouton prediction nous retournerra le type de fleur.
Un onglet input Data nous retournera le dataset, un onglet console nous retournera la precision de l'evaluation ainsi qu'une matrice de confusion.
Un onglet de consultation Web de DL4J ou nous pourrons consulter plusieurs informations.


# Creation de l'application

## Initialisation du projet

* Creation d'un projet Maven


```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	<modelVersion>4.0.0</modelVersion>
	<groupId>com.ruffin</groupId>
	<artifactId>dl4j</artifactId>
	<version>0.0.1-SNAPSHOT</version>
	<name>Deep Learning for Java</name>
	<properties>
		<maven.compiler.source>15</maven.compiler.source>
		<maven.compiler.target>15</maven.compiler.target>
		<project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
		<project.reporting.outputEncoding>UTF-8</project.reporting.outputEncoding>
	</properties>
	<dependencies>
		<!--Coeur de DL4J -->
		<dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-core</artifactId>
			<version>1.0.0-beta3</version>
		</dependency>
		<!--ND4J Natif pour CPU -->
		<dependency>
			<groupId>org.nd4j</groupId>
			<artifactId>nd4j-native-platform</artifactId>
			<version>1.0.0-beta3</version>
		</dependency>
				<!--ND4J Natif pour GPU -->
		<dependency>
			<groupId>org.nd4j</groupId>
			<artifactId>nd4j-cuda-9.2-platform</artifactId>
			<version>1.0.0-beta3</version>
		</dependency>
		<!--User Interface de DL4J -->
		<dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-ui_2.11</artifactId>
			<version>1.0.0-beta3</version>
		</dependency>
		<dependency>
			<groupId>org.slf4j</groupId>
			<artifactId>slf4j-simple</artifactId>
			<version>1.6.1</version>
		</dependency>
	</dependencies>
</project>
```


* Ajout d'un fichier Readme et gitignore

* Initialisation depot git

```cmd
cd dl4j
git init
git add .
git commit -am "init projet"
```

* Creation d'un repository dans github

* Faire le lien entre le repository local et Github

```cmd
git branch -M main
git remote add origin https://github.com/jeanyvesruffin/DeepLearning4Java.git
git push -u origin main
```

## Creation du modele.

Configuration du modele a l'aide de MLP (MultiLayer Perceptron).

```java
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

MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration
	.Builder()
	.seed(123)
	.updater(new Adam(learningRate))
		.list()
		.layer(0,	new DenseLayer
					.Builder()
					.nIn(inputSize)
					.nOut(numHiddenNodes)
					.activation(Activation.SIGMOID)
					.build()
				)
		.layer(1,	new OutputLayer
					.Builder()
					.nIn(numHiddenNodes)
					.nOut(outputSize)
					.lossFunction(LossFunctions
									.LossFunction
									.MEAN_SQUARED_LOGARITHMIC_ERROR
									)
					.activation(Activation.SOFTMAX)
					.build()
				)
		.build();

// Creation du modele
MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
multiLayerNetwork.init();

```

## Lecture des donnees a partir du dataset

DataVect possede des methodes permettant de lire des fichiers csv.

Le batchSize permet de dimensionner le nombre de ligne, du jeu de donnees, qui seront utilisees pour faire le calcule des poids des connexions neuronnale.

Si batchSize = 1 alors l'algorithme d'entrainement prendra chaque ligne du fichier CSV qui seront analysees puis passera a la suivante afin de minimiser l'erreur.


```java

// Lecture du fichier CSV et Entrainement du modele
System.out.println("Entrainement du modele");

File fileTrain = new ClassPathResource("iris-train.csv").getFile();
RecordReader recordReaderTrain = new CSVRecordReader();
recordReaderTrain.initialize(new FileSplit(fileTrain));

// organisation dans un dataset Iterator
int batchSize = 1;
int classIndex = 4;

DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex,outputSize);

```

Creation et visualisation du dataSetIterator

```java

// Lecture du fichier CSV et Entrainement du modele
System.out.println("Entrainement du modele");
File fileTrain = new ClassPathResource("iris-train.csv").getFile();
RecordReader recordReaderTrain = new CSVRecordReader();
recordReaderTrain.initialize(new FileSplit(fileTrain));
// organisation dans un dataset Iterator
int batchSize = 1;
int classIndex = 4;
DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex,
		outputSize);
// Parcour du dataset batch par batch

while(dataSetIteratorTrain.hasNext()) {
	System.out.println("------------------------------------------------------------");
	DataSet dataSet = dataSetIteratorTrain.next();
	System.out.println(dataSet.getFeatures());
	System.out.println(dataSet.getLabels());
}
```

```java
Entrainement du modele
------------------------------------------------------------
[[    5.1000,    3.5000,    1.4000,    0.2000]]
[[    1.0000,         0,         0]]
```

Cette matrice correspondant a la 1er feature (1er ligne) ayant en entree les valeurs 5.1, 3.5, 1.4 et 0.2 et en sortie 1,0,0 (cela represente l'etat des sortie, ici le 1 correspond au 1er type d'iris..)

Si l'on change la taille du batchSize a 30 vs 1 nous pouvons constater que l'organisation du dataSet sera groupe par 30.

## Entrainement du modele

Le nombre d'epochs correspond au nombre de fois que l'entrainement vas etre effectue (ici 500 fois). cela correspond au cycle d'apprentissage pour optimisation.

Pour controler l'evolution de l'apprentissage nous pouvons utiliser DL4J avec UIserver.

```java
// Demarrage du serveur de monitoring du processus d'apprentissage avec UIServer utilise le port 9000 par defaut
UIServer uiServer=UIServer.getInstance();
InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
uiServer.attach(inMemoryStatsStorage);

multiLayerNetwork.setListeners(new StatsListener(inMemoryStatsStorage));

int numEpochs = 500;
for (int i = 0; i < numEpochs; i++) {
	multiLayerNetwork.fit(dataSetIteratorTrain);
}
```

## Evaluation du modele

```java
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
```
```cmd

========================Evaluation Metrics========================
 # of classes:    3
 Accuracy:        1,0000
 Precision:       1,0000
 Recall:          1,0000
 F1 Score:        1,0000
Precision, recall & F1: macro-averaged (equally weighted avg. of 3 classes)


=========================Confusion Matrix=========================
 0 1 2
-------
10 0 0 | 0 = 0
 010 0 | 1 = 1
 0 010 | 2 = 2

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

Cette evaluation nous montre que nos predictions sont correctes à 100%.

Si l'on change le nombre d'epochs que l'on passe a 30.

```cmd

========================Evaluation Metrics========================
 # of classes:    3
 Accuracy:        0,8333
 Precision:       0,8333
 Recall:          0,8889
 F1 Score:        0,8222
Precision, recall & F1: macro-averaged (equally weighted avg. of 3 classes)


=========================Confusion Matrix=========================
 0 1 2
-------
10 0 0 | 0 = 0
 0 5 0 | 1 = 1
 0 510 | 2 = 2

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

Nous pouvons constater alors que nos predictions sont correcte a 83% seulement. 
Pour la classe 2 la moitier est correct l'utre se trompe pour etre sur le type 3.

## Predictions pour une donnee jamais connue


# Execution du modele et utilisation de l'interface DL4J

Executer votre application est appeler dans votre navigateur l'adresse url localhost:9000

<img src="src/main/resources/images/ServerUI.bmp">

<img src="src/main/resources/images/ServerUI_2.bmp">
 