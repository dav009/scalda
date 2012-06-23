package edu.ucla.sspace.graphical

import DistanceMetrics.euclidean

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Gamma


class SphericalGaussianRasmussen(val mu_0: DenseVectorRow[Double],
                                 val variance_0: Double) extends ComponentGenerator {
    val v = mu_0.length
    val sigma_0 = sqrt(variance_0)
    var lambda = DenseVectorRow.randn(v) :* sigma_0 :+ mu_0
    var rho = new Gamma(1, 1/variance_0).sample
    var beta = new Gamma(1, 1).sample
    var omega = new Gamma(1, variance_0).sample

    def initial() =
        (0d, initialMean, initialVariance)
    def initialMean() = 
        DenseVectorRow.randn(v) :* sqrt(1/rho) :+ lambda
    def initialVariance() = 
        1/sampleGamma(beta, 1/omega)

    def sample(data: List[VectorRow[Double]], sigma_2_prime: Double) = {
        val rho_prime = 1/sigma_2_prime
        val variance_prior = 1 / (data.size * rho_prime + rho)
        val sigma_prior = sqrt(variance_prior)
        val mu_prime = data.reduce(_+_) / data.size
        val mu_prior = (mu_prime * data.size * rho_prime + lambda * rho) *
                       variance_prior
        var mu = DenseVectorRow.randn(lambda.length) :* sigma_prior :+ mu_prior

        val beta_prior = beta + data.size
        val omega_prior = beta_prior/(omega*beta + data.size*computeVariance(data, mu))
        val sigma_2 = 1/sampleGamma(beta_prior, omega_prior)
        (data.size.toDouble, mu, sigma_2)
    }

    def update(mu_k: Array[DenseVectorRow[Double]], variance_k: Array[Double]) {
        val k = mu_k.size
        lambda = sampleLambda(mu_k, k)
        rho = sampleRho(mu_k, k)
        omega = sampleOmega(variance_k, k)
        beta = sampleBeta(variance_k, k)
    }

    def printMean(mu: DenseVectorRow[Double]) {
        println(mu.pairsIterator.filter(_._2 > 0d).map{ case(k,v) => k + ":" + v}.mkString(" "))
    }

    def sampleLambda(mu_k: Array[DenseVectorRow[Double]], k: Int) = {
        val variance_prior = 1/(1/variance_0 + k*rho)
        val mu_prior = ((mu_0 / variance_0) :+ (mu_k.reduce(_+_) * rho))*variance_prior
        val sigma_prior = sqrt(variance_prior)
        DenseVectorRow.randn(v) :* sigma_prior :+ mu_prior
    }

    def sampleRho(mu_k: Array[DenseVectorRow[Double]], k: Int) = {
        val beta_prior = k+1
        val omega_prior = beta_prior / (variance_0 + k*computeVariance(mu_k, lambda))
        sampleGamma(beta_prior, omega_prior)
    }

    def sampleOmega(variance_k: Array[Double], k: Int) = {
        val beta_prior = k*beta + 1
        val omega_prior = beta_prior / (1/variance_0 + beta*variance_k.map(1/_).sum)
        sampleGamma(beta_prior, omega_prior)
    }

    def sampleBeta(variance_k: Array[Double], k: Int) = {
        beta
    }

    def computeVariance(data: Seq[VectorRow[Double]],
                        mu: DenseVectorRow[Double]) = 
        data.map(euclidean(mu, _))
            .map(pow(_, 2))
            .reduce(_+_) / data.size

    def sampleGamma(beta: Double, omega: Double) =
        new Gamma(beta/2d, 2*omega/beta).sample
}
