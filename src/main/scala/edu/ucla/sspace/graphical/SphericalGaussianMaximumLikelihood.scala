package edu.ucla.sspace.graphical

import scalala.library.Library.sqrt
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow


class SphericalGaussianMaximumLikelihood(val mu_0: DenseVectorRow[Double],
                                         val variance_0: Double) extends ComponentGenerator {

    val v = mu_0.length
    val sigma_0 = sqrt(1/variance_0)

    def initial() =
        (0d, initialMean, initialVariance)
    def initialMean() = 
        DenseVectorRow.randn(v) :* sigma_0 :+ mu_0
    def initialVariance() = 
        variance_0

    def sample(data: List[VectorRow[Double]], sigma_2_old: Double) = {
        val n = data.size.toDouble
        val mu = data.reduce(_+_).toDense / n
        val sigma = if (data.size == 1) sigma_2_old else data.map(_-mu).map(_.norm(2)).sum/n
        (n, mu, sigma)
    }

    def update(mu_k: Array[DenseVectorRow[Double]], variance_k: Array[Double]) {
    }
}
