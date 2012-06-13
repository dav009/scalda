package edu.ucla.sspace.graphical

import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow


trait ComponentGenerator {
    def initial() : (Double, DenseVectorRow[Double], Double)
    def sample(data: List[VectorRow[Double]], sigma_2_old: Double) : (Double, DenseVectorRow[Double], Double)
    def update(mu_k: Array[DenseVectorRow[Double]], variance_k: Array[Double])
}
