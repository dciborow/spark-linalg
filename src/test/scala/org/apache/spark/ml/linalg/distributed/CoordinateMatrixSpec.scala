package org.apache.spark.ml.linalg.distributed

import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions.rand
import org.scalatest.FunSuite

class CoordinateMatrixSpec extends FunSuite {
  private[spark] val conf: SparkConf = new SparkConf()
    .setAppName("Testing Recommendation Model")
    .setMaster("local[*]")
    .set("spark.driver.memory", "80g")
    .set("spark.driver.maxResultSize", "0")

  lazy val session: SparkSession = SparkSession.builder()
    .config(conf)
    .getOrCreate()
  session.sparkContext.setLogLevel("WARN")
  /**
    * Test multiplication of...
    * ...two square matrices
    * ...one rectangular, and one square matrix
    */
  test("Mat Mul test") {
    def testMatrixMultiplication(row: Int, col: Int, pctDense: Double): Unit = {
      implicit def dsToSM(ds: Dataset[MatrixEntry]): CoordinateMatrix = new CoordinateMatrix(ds)

      val xds = makeMatrixDF(row, col, pctDense)
      assert(xds.numRows() == row)
      assert(xds.numCols() == col)

      val yds = makeMatrixDF(col, col, pctDense)
      assert(yds.numCols() == col)
      assert(yds.numRows() == col)

      val product = xds multiply yds
      assert(product.numCols() == col)
      assert(product.numRows() == row)

      val rddProduct = xds.toMLLibBlockMatrix multiply yds.toMLLibBlockMatrix
      assert(product.toMLLibLocalMatrix == rddProduct.toLocalMatrix().asML)
      println(product.entries.collect)
      //      assert(product.toBlockMatrix().toLocalMatrix() == rddProduct.toLocalMatrix())
      ()
    }
    testMatrixMultiplication(3, 3, 1)
    testMatrixMultiplication(30, 30, .9)
    testMatrixMultiplication(1000, 1000, .1)
  }

  def makeMatrixDF(rowCount: Int, colCount: Int, pctDense: Double = 0.2): Dataset[MatrixEntry] = {
    import session.implicits._

    val rows = session.sqlContext.range(0, rowCount.toLong)

    val cols = session.sqlContext.range(0, colCount.toLong)

    rows
      .crossJoin(cols)
      .withColumn("rand", rand(5))
      .map(row => MatrixEntry(row.getLong(0), row.getLong(1), row.getDouble(2)))
      .sample(false, pctDense)
  }
}
