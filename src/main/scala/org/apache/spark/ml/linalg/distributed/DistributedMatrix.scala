// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.linalg.distributed

/**
  * Represents a distributively stored matrix backed by one or more DataSets.
  */
trait DistributedMatrix extends Serializable {

  /** Gets or computes the number of rows. */
  def numRows(): Long

  /** Gets or computes the number of columns. */
  def numCols(): Long

}
