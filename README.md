
## Background
The current matrix multiplication implementation in Spark runs into scaling issues. The matrix multiplcaiton plan becomes unrunable when the number of columns in the first matrix exceeds 40,000. This new appraoch is designed to overcome that issue, and to provide a more scalable implmentation of matrix multiplication that can complete when matrix sizes exceed millions of columns and rows. A script is included, SparseSparseRDDTimer, that can be used to easily reproduce the observed issues with the current matrix multiplication implementation.  


## Benchmarks
Benchmark results can be found in Benchmarks.xlsx, and the code to generate the benchmark results is included in Benchmarks.scala. The benchmarks were run on Azure Databricks and the cluster configurations can be found in each tab. For larger benchmarks, it is recommended to first generate and save the dataset, and to also write the dataset to disc. This is to avoid collecting the results onto the driver, while forcing Spark's lazy execution. Sample code for this is included in DataGeneration.scala.   

## Setup

###### Downloads
First, ensure that both Java 1.8 and Maven are installed. Newer versions of Java may cause build issues.

Install java 1.8 in an easily accessible directory (for this example,  ~/java/)
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

 
Download apache Maven 3.3.9 and un-tar/gunzip to ~/apache/apache-maven-3.3.9/ .
https://maven.apache.org/download.cgi

Create a directory ~/src/ .   

```
mkdir ~/src/
cd ~/src/
```

Clone project repository into `~/src`.

```
git clone https://github.com/dciborow/spark-linalg.git
```    
 
###### Building Project with Apache Maven
From the  project directory we may issue the command to build using the mvn profile.

JVM build:
```
mvn clean install -DskipTests
```

#### Testing the Project Environment

To launch the shell in local mode with 2 threads: simply do the following:
```
$ spark-shell MASTER=local[2] --jars="target/linalg-1.0-SNAPSHOT.jar"
```

At the scala> prompt, enter: 
```   
scala> :load examples/bin/SparseSparseDrmTimer.scala
```
Which will load a matrix multiplication timer function definition. To run the matrix timer: 
```
scala> timeSparseDRMMMul(1000,1000,1000,1,.02,1234L)
    {...} res3: Long = 16321
```

## Azure Databricks - Setup

First, follow the above directions to use Maven to build a JAR from the project. Then, upload the jar as a new library in Databricks. Attach this library to your desired cluster. Next, upload Eenchmarks.scala. Run this notebook on your attached cluster. 

## Troubleshooting

If you run into the following error, you may have set the sample ratio to small compared to the size of your matrices. 
Try increasing the sample ratio. 
```
java.lang.UnsupportedOperationException: empty collection
```