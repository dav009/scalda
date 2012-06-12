package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.ComponentGenerator
import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Likelihood._
import edu.ucla.sspace.graphical.Util.{norm,epsilon}

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Multinomial
import scalanlp.stats.distributions.Gamma

import scala.util.Random


class FiniteSphericalGaussianMixtureModel(val numIterations: Int, 
                                          val alpha: Double,
                                          val generator: ComponentGenerator,
                                          s: Set[Int] = Set[Int]()) extends Learner {

    type Theta = (Double, DenseVectorRow[Double], Double)

    def train(data: List[VectorRow[Double]], k: Int) = {
        // Extract the shape of the data.
        val n = data.size
        val v = data(0).length

        val alpha_k = alpha/k.toDouble

        val mu_0 = data.reduce(_+_).toDense / n
        val variance_0 = variance(data, mu_0)
        val sigma_0 = sqrt(variance_0)

        var lambda = DenseVectorRow.randn(v) :* sigma_0 :+ mu_0
        var rho = new Gamma(1, 1/variance_0).sample
        var beta = new Gamma(1, 1).sample
        var omega = new Gamma(1, variance_0).sample

        val labels = Array.fill(n)(0)
        var components = Array.fill(k)(initialComponent(lambda, rho, beta, omega))

        for ((x_j,j) <- data.zipWithIndex) {
            def dataLikelihood(theta: Theta) = gaussian(x_j, theta._2, theta._3)
            val likelihood = DenseVectorRow[Double](components.map(dataLikelihood))
            val l_j = likelihood.argmax
            components(l_j) = updateComponent(components(l_j), x_j, +1)
            labels(j) = l_j
        }
        if (s contains 0)
            report(0, labels.toList)

        components = updateComponents(components, labels, data, lambda, rho, beta, omega)

        components.foreach(println)
        val mu_k = components.map(_._2)
        val variance_k = components.map(_._3)
        lambda = sampleLambda(mu_0, mu_k, variance_0, rho, k)
        rho = sampleRho(k, variance_0, mu_k, lambda)
        omega = sampleOmega(k, beta, variance_0, variance_k)
        beta = sampleBeta(beta, k, variance_k, omega)

        def priorLikelihood(theta: Theta) = (theta._1 + alpha_k)/(n-1+alpha)

        for (i <- 0 until numIterations) {
            printf("Iteration [%d]\n", i)
            for ( (x_j, j) <- data.zipWithIndex ) {
                def dataLikelihood(theta: Theta) = gaussian(x_j, theta._2, theta._3)

                val l_j = labels(j)
                components(l_j) = updateComponent(components(l_j), x_j, -1)

                val prior = DenseVectorRow[Double](components.map(priorLikelihood))
                val likelihood = DenseVectorRow[Double](components.map(dataLikelihood))
                val probs = norm(prior :* likelihood)

                val l_j_new  = new Multinomial(probs).sample
                components(l_j_new) = updateComponent(components(l_j_new), x_j, +1)
                labels(j) = l_j_new
            }

            components = updateComponents(components, labels, data, lambda, rho, beta, omega)

            val mu_k = components.map(_._2)
            val variance_k = components.map(_._3)
            lambda = sampleLambda(mu_0, mu_k, variance_0, rho, k)
            rho = sampleRho(k, variance_0, mu_k, lambda)
            omega = sampleOmega(k, beta, variance_0, variance_k)

            components.foreach(println)

            if (s contains (i+1))
                report(i+1, labels.toList)
        }

        labels
    }

    def variance(data: Seq[VectorRow[Double]],
                 mu: DenseVectorRow[Double]) = 
        data.map(_-mu)
            .map(_.norm(2))
            .map(pow(_, 2))
            .foldLeft(0d)(_+_) / data.size

    def initialComponent(lambda: DenseVectorRow[Double],
                         rho: Double,
                         beta: Double,
                         omega: Double) : Theta = {
        val mu = DenseVectorRow.randn(lambda.length) :* sqrt(1/rho) :+ lambda
        val sigma = 1/sampleGamma(beta, 1/omega)
        (0d, mu, sigma)
    }

    def sampleComponent(data: List[VectorRow[Double]],
                        rho_prime: Double,
                        lambda: DenseVectorRow[Double],
                        rho: Double,
                        beta: Double,
                        omega: Double) : Theta = {
        val variance_prior = 1 / (data.size * rho_prime + rho)
        val sigma_prior = sqrt(variance_prior)
        val mu_prime = data.reduce(_+_) / data.size
        val mu_prior = (mu_prime * data.size * rho_prime + lambda * rho) *
                       variance_prior
        var mu = DenseVectorRow.randn(lambda.length) :* sigma_prior :+ mu_prior

        val beta_prior = beta + data.size
        val omega_prior = beta_prior/(omega*beta + data.size*variance(data, mu))
        val sigma_2 = 1/sampleGamma(beta_prior, omega_prior)
        (data.size, mu, sigma_2)
    }

    def sampleLambda(mu: DenseVectorRow[Double],
                     mu_k: Array[DenseVectorRow[Double]],
                     variance: Double,
                     rho: Double,
                     k: Int) = {
        val variance_prior = 1/(1/variance + k*rho)
        val mu_prior = ((mu / variance) :+ (mu_k.reduce(_+_) * rho))*variance_prior
        val sigma_prior = sqrt(variance_prior)
        DenseVectorRow.randn(mu.length) :* sigma_prior :+ mu_prior
    }

    def sampleRho(k: Int,
                  variance_0: Double,
                  mu_k: Array[DenseVectorRow[Double]],
                  lambda: DenseVectorRow[Double]) = {
        val beta_prior = k+1
        val omega_prior = beta_prior / (variance_0 + k*variance(mu_k, lambda))
        new Gamma(beta_prior, omega_prior).sample
    }

    def sampleOmega(k: Int,
                    beta: Double,
                    variance: Double,
                    variance_k: Array[Double]) = {
        val beta_prior = k*beta + 1
        val omega_prior = beta_prior / (1/variance + beta*variance_k.map(1/_).sum)
        new Gamma(beta_prior, omega_prior).sample
    }

    def sampleBeta(beta: Double,
                   k: Int,
                   variance_k: Array[Double],
                   omega: Double) = {
        beta
    }

    def updateComponents(components: Array[Theta],
                         labels: Array[Int],
                         data: List[VectorRow[Double]],
                         lambda: DenseVectorRow[Double],
                         rho: Double,
                         beta: Double,
                         omega: Double) = {
        val groups = labels.zip(data).groupBy(_._1)
                           .map{ case(k,v) => (k, v.map(_._2).toList) }
        components.zipWithIndex.map{ case (theta, c) => 
                groups.get(c) match {
                    case Some(x) => sampleComponent(x, 1/theta._3,lambda, rho, beta, omega)
                    case None => initialComponent(lambda, rho, beta, omega)
                }
        }
    }

    def updateComponent(theta: Theta, x: VectorRow[Double], delta: Double) =
        if (delta >= 0)
            (theta._1+delta, theta._2, theta._3)
        else
            (theta._1+delta, theta._2, theta._3)

    def sampleGamma(beta: Double, omega: Double) =
        new Gamma(beta/2d, 2*omega/beta).sample
}
