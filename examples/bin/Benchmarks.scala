// Databricks notebook source
import org.apache.spark.ml.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.rand


def timeSparseDRMMMul(m: Int, n: Int, s: Int, para: Int, pctDense: Double = .20, seed: Long = 1234L): Long = {

  val session = SparkSession.builder().getOrCreate()

  def makeMatrixDF(rowCount: Int, colCount: Int): CoordinateMatrix = {
    import session.implicits._

    val rows = session.sqlContext.range(0, rowCount.toLong)
    val cols = session.sqlContext.range(0, colCount.toLong)

    val replacement = false
    new CoordinateMatrix(
      rows
        .crossJoin(cols)
        .sample(replacement, pctDense, seed)
        .withColumn("rand", rand(seed))
        .map(row => MatrixEntry(row.getLong(0), row.getLong(1), row.getDouble(2))))
  }

  val left = makeMatrixDF(m, n)
  val right = makeMatrixDF(n, s)

  val start = System.currentTimeMillis()
  val product = left.multiply(right)
  println(product.entries.collect)
  val end = System.currentTimeMillis()
  end - start
}

// COMMAND ----------

timeSparseDRMMMul(1,1,1,1,1)

// COMMAND ----------

timeSparseDRMMMul(10,10,1,1,1)

// COMMAND ----------

timeSparseDRMMMul(10,10,10,1,1)

// COMMAND ----------

timeSparseDRMMMul(100,100,1,1,1)

// COMMAND ----------

timeSparseDRMMMul(100,100,10,1,1)

// COMMAND ----------

timeSparseDRMMMul(100,100,100,1,1)

// COMMAND ----------

timeSparseDRMMMul(1000,1000,1,1,1)

// COMMAND ----------

timeSparseDRMMMul(1000,1000,10,1,1)

// COMMAND ----------

timeSparseDRMMMul(1000,1000,100,1,1)

// COMMAND ----------

timeSparseDRMMMul(1000,1000,1000,1,1)

// COMMAND ----------

timeSparseDRMMMul(10000,10000,1,1,1)

// COMMAND ----------

timeSparseDRMMMul(10000,10000,10,1,1)

// COMMAND ----------

timeSparseDRMMMul(10000,10000,100,1,1)

// COMMAND ----------

timeSparseDRMMMul(10000,10000,1000,1,1)

// COMMAND ----------

timeSparseDRMMMul(10000,10000,10000,1,1)

// COMMAND ----------


