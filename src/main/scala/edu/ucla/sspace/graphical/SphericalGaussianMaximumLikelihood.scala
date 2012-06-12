package edu.ucla.sspace.graphical

import scalala.tensor.mutable.VectorRow


class SphericalGaussianMaximumLikelihood extends ComponentGenerator {

    def sample(data: List[VectorRow[Double]], sigma_2_old: Double) = {
        val n = data.size.toDouble
        val mu = data.reduce(_+_).toDense / n
        val sigma = if (data.size == 1) sigma_2_old else data.map(_-mu).map(_.norm(2)).sum/n
        (n, mu, sigma)
    }
}
