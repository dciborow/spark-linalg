
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

JVM only:
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
scala> :load ../msft-linalg/examples
                               /bin/SparseSparseDrmTimer.scala
```
Which will load a matrix multiplication timer function definition. To run the matrix timer: 
```
        scala> timeSparseDRMMMul(1000,1000,1000,1,.02,1234L)
            {...} res3: Long = 16321
```

#### Troubleshooting

If you run into the following error, you may have set the sample ratio to small compared to the size of your matrices. 
Try increasing sample ratio. 
```
java.lang.UnsupportedOperationException: empty collection
```