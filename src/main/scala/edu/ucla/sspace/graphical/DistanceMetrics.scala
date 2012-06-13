package edu.ucla.sspace.graphical

import scalala.library.Library._
import scalala.tensor.VectorRow
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.sparse.SparseVectorRow


object DistanceMetrics {
    def euclidean(v1: VectorRow[Double], v2: VectorRow[Double]) : Double =
        (v1, v2) match {
            case (dv1: DenseVectorRow[Double], dv2: DenseVectorRow[Double]) => euclidean(dv1, dv2)
            case (dv1: DenseVectorRow[Double], sv2: SparseVectorRow[Double]) => euclidean(dv1, sv2)
            case (sv2: SparseVectorRow[Double], dv1: DenseVectorRow[Double]) => euclidean(dv1, sv2)
            case (sv1: SparseVectorRow[Double], sv2: SparseVectorRow[Double]) => euclidean(sv1, sv2)
        }

    def euclidean(v1: DenseVectorRow[Double], v2: DenseVectorRow[Double]) : Double =
        sqrt(((v1 - v2) :^ 2).sum)

    def euclidean(v1: VectorRow[Double], v2: SparseVectorRow[Double]) : Double = {
        var v1Magnitude = pow(v1.norm(2), 2)
        val dist = v2.mapNonZeroPairs( (i, y) => {
            val v1Val = v1(i)
            v1Magnitude -= v1Val * v1Val
            pow(y - v1Val, 2)
        }).sum
        sqrt(v1Magnitude + dist)
    }
}
