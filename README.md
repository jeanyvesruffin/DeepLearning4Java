# Deep Learning For Java (DL4J)

## Definition

**Deep Learning 4 java** est un framework open source (licence apache) qui permet de construire, entrainer et tester une grande diversite d'algorithmes de Deep Learning (depuis les reseaux standard, jusqu'aux reseau concolutionels, en passant par des architectures plus complexes.

Il se base sur sa structure de donnees (Nd4j) permettant d'effectuer les operations de l'algebres lineaires sur les architectures massivement paralleles GPU et les architectures distribuees.

**Nd4j** utilise du code natif (Cuda), et alloue de l'espace or du tas Java. Ceci est imperativement a prendre en compte lorsque la volumetrie des donnees est importante.

DL4J utilise **DataVec** pour la vectorisation et la transformation des donnes.

## But de l'application

Le but est de creer un modele d'entrainement supervise a l'aide du dataset Iris. Le dataset fournis 120 instances (120 lignes) d'exemples de donnees de fleurs, classees en 3 types de fleurs (iris-setosa, iris-versicolor et iris virginica) et pour chaque type de fleurs nous avons 4 carateristiques (SepalLength, SepalWidth, PetalLength et PetalWidth)

<img src="src/main/resources/images/Iris-setosa.png"  width="120" height="120">
<img src="src/main/resources/images/Iris-versicolor.png"  width="120" height="120">
<img src="src/main/resources/images/Iris-virginica.png"  width="120" height="120">



# Initialisation du projet

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


