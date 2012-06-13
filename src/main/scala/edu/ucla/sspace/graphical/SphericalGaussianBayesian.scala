package edu.ucla.sspace.graphical

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Gamma


class SphericalGaussianBayesian(val mu_0: DenseVectorRow[Double],
                                val variance_0: Double,
                                val alpha_prior: Double,
                                val beta_prior: Double) extends ComponentGenerator {
    val v = mu_0.length
    val sigma_0 = sqrt(1/variance_0)
    val rho_prior = 1 / variance_0
    val mu_rho_prior = mu_0 * rho_prior

    def initial() =
        (0d, DenseVectorRow.randn(v) :* sigma_0 :+ mu_0, variance_0)

    def sample(data: List[VectorRow[Double]], sigma_2_old: Double) = {
        val sigma_2_hyper = 1/(rho_prior + (data.size/ sigma_2_old))
        val sigma_hyper = sqrt(sigma_2_hyper)
        var mu_hyper = (mu_rho_prior + (data.reduce(_+_).toDense / sigma_2_old)) *
                       sigma_2_hyper
        val mu_prime = DenseVectorRow.randn(mu_rho_prior.length)
        val mu = mu_prime :* sigma_hyper :+ mu_hyper

        val alpha = alpha_prior + data.size / 2d
        val beta = beta_prior + data.map(_-mu)
                                 .map(_.norm(2))
                                 .map(pow(_,2))
                                 .reduce(_+_)/2d
        val sigma_2 = 1 / new Gamma(alpha, 1/beta).sample
        (data.size, mu, sigma_2)
    }

    def update(mu_k: Array[DenseVectorRow[Double]], variance_k: Array[Double]) {
    }
}
