// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.ml.linalg.{Matrix, Vectors => V}
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix => CM}
import org.apache.spark.sql.functions.{col, collect_list, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable
import scala.language.implicitConversions

/**
  * Sparse Matrix wrapped for Spark Dataset of MatrixEntry
  *
  * Based on the requirements for Spark's CoordinateMatrix
  * An RDD that would be used to create a CoordinateMatrix, can be used to create a
  * Sparse Matrix using rdd.toDS when it is an RDD of MatrixEntry.
  *
  * implicit def dsToSM(ds: Dataset[MatrixEntry]): SparseMatrix = new spark.linalg.SparseMatrix(ds)
  * val rddMatrixEntry: RDD[MatrixEntry] = rdd.map(row => new MatrixEntry(row._1, row._2, row._3)
  * val sparseMatrix: SparseMatrix = rddMatrixEntry.toDS
  *
  * Or, a DataFrame of Row, can be mapped to a MatrixEntry.
  *
  * implicit def dsToSM(ds: Dataset[MatrixEntry]): SparseMatrix = new spark.linalg.SparseMatrix(ds)
  * val sparseMatrix: SparseMatrix =
  *     dataframe.map(row => new MatrixEntry(row.getLong(0). row.getLong(1), row.getDouble(2))
  *
  * The SparseMatrix can be easily converted back to the MLlib distributed matricies using the same
  * type of casting methods.
  *
  * val coordinateMatrix: CoordinateMatrix = sparseMatrix.toCoordinateMatrix
  *
  * val blockMatrix: BlockMatrix = sparseMatrix.toBlockMatrix
  *
  * val localMatrix: LocalMatrix = sparseMatrix.toLocalMatrix
  *
  * @param entries Input data, a Dataset of MatrixEntry
  * @param nRows   number of rows, in case padding is needed
  * @param nCols   number of columns, in case padding is needed
  */
class CoordinateMatrix(val entries: Dataset[MatrixEntry],
                       private var nRows: Long,
                       private var nCols: Long) extends DistributedMatrix with MLLib {

  private[ml] def toBreeze(): BDM[Double] = {
    val m = numRows().toInt
    val n = numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    entries.collect().foreach { case MatrixEntry(i, j, value) =>
      mat(i.toInt, j.toInt) = value
    }
    mat
  }

  //  def multiply(other: IndexedRowMatrix): DistributedMatrix = multiply(other.toCoordinateMatrix)

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  def this(entries: Dataset[MatrixEntry]) = this(entries, 0L, 0L)

  /**
    * Get Number of Columns in Matrix
    *
    * @return
    */
  override def numCols(): Long = {
    if (nCols <= 0L) {
      computeSize()
    }
    nCols
  }

  /**
    * Get Number of Rows in Matrix
    *
    * @return
    */
  override def numRows(): Long = {
    if (nRows <= 0L) {
      computeSize()
    }
    nRows
  }

  /**
    * Transpose Matrix
    *
    * @return
    */
  def transpose(): CoordinateMatrix = {
    import entries.sparkSession.implicits._
    new CoordinateMatrix(
      entries.mapPartitions(_.map(x => MatrixEntry(x.j, x.i, x.value))),
      nCols, nRows)
  }

  /**
    * Compute the Rows and Columns of the Matrix
    * Used by the cols and rows getters
    */
  private def computeSize(): Unit = {
    // There may be empty columns at the very right and empty rows at the very bottom.
    import entries.sparkSession.implicits._
    val (m1, n1) = entries
      .map(entry => (entry.i, entry.j))
      .reduce((l1, l2) => (math.max(l1._1, l2._1), math.max(l1._2, l2._2)))

    nRows = math.max(nRows, m1 + 1L)
    nCols = math.max(nCols, n1 + 1L)
  }

  /**
    * Multiply two sparse matrices together
    *
    * @param other left matrix for multiplication, in SparseMatrix type
    * @return Product of A * B
    */
  def multiply(other: CoordinateMatrix): CoordinateMatrix = multiply2(this, other)

  /**
    * Multiply two sparse matrices together
    *
    * @param other left matrix for multiplication
    * @return Product of A * B
    */
  def multiply(other: Dataset[MatrixEntry]): CoordinateMatrix = {
    multiply2(new CoordinateMatrix(entries), new CoordinateMatrix(other))
  }

  case class VectorPair(i: Int, iVec: mutable.WrappedArray[(Int, Double)], j: Int, jVec: mutable.WrappedArray[(Int,
    Double)])

  def test: Unit = {
    val spark = SparkSession.builder().getOrCreate()
    // With some step, square matrices and multiple runs
    val minDim = 1000
    val maxDim = 5000
    val step = 1000
    val repetitions = 5
    val range = for (b <- minDim to maxDim by step) yield b
    //val range = for (b  <- 6 to 14) yield { math.pow(2, b).toInt}
    import spark.implicits._

    for (i <- 0 until range.length) {
      var result = new Array[Double](repetitions)
      val dim = range(i)
      for (j <- 0 until repetitions) {
        //        val a = breeze.linalg.DenseMatrix.rand[Double](dim, dim)//.mapValues(_.toFloat);
        //        val b = breeze.linalg.DenseMatrix.rand[Double](dim, dim)//.mapValues(_.toFloat);
        //        val aMat = Matrices.dense(a.rows, a.cols, a.toArray)
        //        val bMat = Matrices.dense(b.rows, b.cols, b.toArray)
        val df = spark.sparkContext.parallelize(List.range(1, 10000)).toDF().select("value")
        System.gc();
        val t = System.nanoTime();
        val product = df.rdd.mapPartitions(rows => {
          rows.map(row => {
            val a = breeze.linalg.DenseMatrix.rand[Double](dim, dim)//.mapValues(_.toFloat);
            val b = breeze.linalg.DenseMatrix.rand[Double](dim, dim)//.mapValues(_.toFloat);
            a.toDenseVector*b.toDenseVector
          })
        })
        println(product.collect)

        //        val product = ds.mapPartitions(rows => {
        //          rows.map( (row) => {
        //            val mat1 = new breeze.linalg.DenseMatrix[Double](row._1.numRows, row._1.numCols, row._1.toArray)
        //            val mat2 = new breeze.linalg.DenseMatrix[Double](row._2.numRows, row._2.numCols, row._2.toArray)
        //            mat1 * mat2
        //          })
        //        })
        val totalTime = System.nanoTime() - t;
        result(j) = totalTime;
      }
      val someValue = result.last
      scala.util.Sorting.quickSort(result)
      val median = (result(repetitions / 2) + result(repetitions / 2 - 1)) / 2
      val avg = result.sum / result.length
      val gigaFlops = 2.0 / median * dim * dim * dim
      println(dim + "\t" + median / 1e9 + "\t" + avg / 1e9 + "\t" + someValue / 1e9 + "\t" + gigaFlops)
    }
  }

  /**
    * Multiplication of two dataset based matrices
    *
    * @param leftMatrix  the rigth matrix in the multiplcation
    * @param rightMatrix the left matrxi in the multiplication
    * @return
    */
  private[linalg] def multiply2(leftMatrix: CoordinateMatrix, rightMatrix: CoordinateMatrix)
  : CoordinateMatrix = {

    import leftMatrix.entries.sparkSession.implicits._

    val sparseVectorLength = leftMatrix.numCols().toInt

    val dot = udf((iVec: mutable.WrappedArray[Row], jVec: mutable.WrappedArray[Row]) => {
      val iVecSeq = iVec.map(r => (r.getInt(0), r.getDouble(1)))
      val iMLSparse = V.sparse(sparseVectorLength, iVecSeq).toSparse
      val ibsv = new BSV[Double](iMLSparse.indices, iMLSparse.values, iMLSparse.size)

      val jVecSeq = jVec.map(r => (r.getInt(0), r.getDouble(1)))

      val jbdv = new BDV[Double](V.sparse(sparseVectorLength, jVecSeq).toArray)

      (jbdv.asDenseMatrix * ibsv.asCscRow).data(0)
    })

    val rightMatrixVectors = rightMatrix.vectorizeCols("j", "jVec")

    new CoordinateMatrix(
      leftMatrix.vectorizeRows("i", "iVec")
        .crossJoin(rightMatrixVectors)
        .withColumn("product", dot(col("iVec"), col("jVec")))
        .map(row =>
          MatrixEntry(
            row.getAs[Long]("i"),
            row.getAs[Long]("j"),
            row.getAs[Double]("product"))))
  }

  private[distributed] def vectorizeRows(indexName: String, colName: String, itemCount: Int) = {
    import entries.sparkSession.implicits._
    entries
      .groupByKey(_.i)
      .mapGroups((a, b) => {
        val vecSeq = b.map(matEntry => (matEntry.j.toInt, matEntry.value)).toSeq
        (a, V.sparse(itemCount, vecSeq).compressed)
      })
      .withColumnRenamed("_1", indexName)
      .withColumnRenamed("_2", colName)
  }

  private[distributed] def vectorizeCols(indexName: String, colName: String, itemCount: Int) = {
    import entries.sparkSession.implicits._
    entries
      .groupByKey(_.j)
      .mapGroups((a, b) => {
        val vecSeq = b.map(matEntry => (matEntry.i.toInt, matEntry.value)).toSeq
        (a, V.sparse(itemCount, vecSeq).compressed)
      })
      .withColumnRenamed("_1", indexName)
      .withColumnRenamed("_2", colName)
  }

  private[distributed] def vectorizeRows(indexName: String, colName: String): DataFrame = {
    import entries.sparkSession.implicits._
    entries
      .map(d => (d.i, (d.j.toInt, d.value)))
      .groupBy($"_1")
      .agg(collect_list($"_2"))
      .withColumnRenamed("_1", "i")
      .withColumnRenamed("collect_list(_2)", colName)
  }

  private[distributed] def vectorizeCols(indexName: String, colName: String): DataFrame = {
    import entries.sparkSession.implicits._
    entries
      .map(d => (d.j, (d.i.toInt, d.value)))
      .groupBy($"_1")
      .agg(collect_list($"_2"))
      .withColumnRenamed("_1", "j")
      .withColumnRenamed("collect_list(_2)", colName)
  }

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.CoordinateMatrix backed by RDD
    *
    * @return
    */
  private[distributed] override def toMLLib: CM = toMLLibCoordinateMatrix(entries)
}

private[distributed] trait MLLib {

  private[distributed] def toMLLib: CM = ???

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.CoordinateMatrix backed by RDD
    *
    * @return
    */
  private[distributed] def toMLLibCoordinateMatrix(entries: Dataset[MatrixEntry]): CM = new CM(entries.rdd)

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.BlockMatrix backed by RDD
    *
    * Takes parameters to specific the row and column block size
    *
    * @param rowsPerBlock number of items to be placed in each row block
    * @param colsPerBlock number of items to be placed in each column block
    * @return
    */
  private[distributed] def toMLLibBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): mllib.linalg.distributed
  .BlockMatrix = {
    toMLLib.toBlockMatrix(rowsPerBlock, colsPerBlock)
  }

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.BlockMatrix backed by RDD
    *
    * Uses default block size, 1024, for columns and rows
    *
    * @return
    */
  private[distributed] def toMLLibBlockMatrix: mllib.linalg.distributed.BlockMatrix = toMLLibBlockMatrix(1024, 1024)

  /**
    * Convert from Sparse Matrix (backed by DataSet) to a mllib.LocalMatrix backed by RDD
    *
    * @return
    */
  private[distributed] def toMLLibLocalMatrix: Matrix = toMLLibBlockMatrix.toLocalMatrix.asML

}
