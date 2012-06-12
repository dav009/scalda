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

//import cern.jet.random.Gamma

import scala.math.{Pi,E}
import scala.util.Random


class InfiniteSphericalGaussianMixtureModel(val numIterations: Int, 
                                            val alpha: Double,
                                            val generator: ComponentGenerator,
                                            s: Set[Int] = Set[Int]()) extends Learner {

    /**
     * Number of points in component,
     * mean
     * variance
     * assigned points
     */
    type Theta = (Double, DenseVectorRow[Double], Double)// List[VectorRow[Double]])

    def train(data: List[VectorRow[Double]], ignored: Int) = {
        // Extract the shape of the data.
        val n = data.size
        val v = data(0).length
        val t = n - 1 + alpha

        // Compute the global mean of all data points.
        val mu_0 = data.reduce(_+_).toDense / n
        // Compute the variance of all data points to the global mean.
        val variance_0 = data.map(_-mu_0).map(_.norm(2)).map(pow(_, 2)).reduce(_+_) / n

        // Create the global component that will be used to determine when a 
        // new component should be sampled.
        //val components = Array((alpha, mu_0, variance_0)).toBuffer
        val components = Array((alpha, mu_0, variance_0)).toBuffer

        // Setup the initial labels for all the data points.  These start off 
        // with no meaningful value.
        var labels = Array.fill(n)(0)

        for (i <- 0 until numIterations) {
            printf("Starting iteration [%d] with [%d] components,\n", i, components.size-1)
            for ( (x_j, j) <- data.zipWithIndex ) {
                // Setup a helper function to compute the likelihood for point x.
                def likelihood(theta: Theta) = gaussian(x_j, theta._2, theta._3)

                val l_j = labels(j)

                // Undo the assignment for the existing point.  This involes
                // first removing the information from the variance vectors,
                // then removing it from the center vector, then finally undoing
                // the count for the component.  Save the original component 
                // data so we can restore it quickly later on.
                if (i != 0)
                    components(l_j) = updateComponent(components(l_j), x_j, -1)

                // Compute the probability of selecting each component based on
                // their sizes.
                val prior = DenseVectorRow[Double](components.map(_._1 / t).toArray)
                // Compute the probability of the data point given each
                // component using the sufficient statistics.
                val posterior = DenseVectorRow[Double](components.map(likelihood).toArray)
                val probs = norm(prior :* posterior)

                // Combine the two probabilities into a single distribution and
                // select a new label for the data point.
                val l_j_new  = new Multinomial(probs).sample

                if (l_j_new == 0) {
                    // If the global component was created, create a new 
                    // component using just the current data point.
                    labels(j) = components.size
                    components.append(generator.sample(List(x_j), 1d))
                } else {
                    // Restore the bookeeping information for this point using the
                    // old assignment.
                    labels(j) = l_j_new
                    components(l_j_new) = updateComponent(components(l_j_new), x_j, 1)
                }
            }

            println("Sampling new components")
            val sigmas = components.map(_._3)

            // Re-estimate the means, counts, and variances for each component.
            // We do this by first grouping the data points based on their
            // assigned component, summing the points assigned to each
            // component, and finally computing the variance of each point from
            // the mean.
            val labelRemap = labels.zip(data)
                                   .groupBy(_._1)
                                   .map{ case(k,v) => (k, v.map(_._2)) }
                                   .zipWithIndex
                                   .map{ case ((c_old, x), c) => {
                components(c+1) = generator.sample(x.toList, sigmas(c_old))
                (c_old, c+1)
            }}
            components.trimEnd(components.size - (labelRemap.size+1))
            components.foreach(println)
            labels = labels.map(labelRemap)

            if (s contains (i+1))
                report(i+1, labels.toList)
        }

        // Return the labels.
        labels.toArray
    }

    /*
    def empericalTheta(mu_rho_0: DenseVectorRow[Double],
                       rho_0: Double,
                       sigma_2_old: Double,
                       x:List[VectorRow[Double]]) = {
        val n = x.size.toDouble
        val mu = x.reduce(_+_).toDense / n
        val sigma = if (x.size == 1) sigma_2_old else x.map(_-mu).map(_.norm(2)).sum/n
        (n, mu, sigma)
    }
       
    def halfBayesTheta(mu_rho_0: DenseVectorRow[Double],
                       rho_0: Double,
                       sigma_2_old: Double,
                       x:List[VectorRow[Double]]) = {
        val mu = sampleMean(mu_rho_0, rho_0, sigma_2_old, x)
        val sigma_2 = x.map(_-mu).map(_.norm(2)).sum / x.size
        (x.size.toDouble, mu, sigma_2)
    } 

    def fullBayesTheta(mu_rho_0: DenseVectorRow[Double],
                       rho_0: Double,
                       sigma_2_old: Double,
                       x:List[VectorRow[Double]]) = {
        val mu = sampleMean(mu_rho_0, rho_0, sigma_2_old, x)
        val beta_prior = beta + x.size / 2d
        val gamma_prior = x.map(_-mu).map(_.norm(2)).map(pow(_,2)).reduce(_+_)/2d + gamma
        //val sigma_2 = 1d/Gamma.staticNextDouble(beta_prior, gamma_prior)
        println(beta_prior)
        println(gamma_prior)
        val sigma_2 = 1d/(new Gamma(beta_prior, gamma_prior).sample)
        (x.size.toDouble, mu, sigma_2)
    }

    def sampleMean(mu_rho_0: DenseVectorRow[Double],
                   rho_0: Double,
                   sigma_2_old: Double,
                   x:List[VectorRow[Double]]) = {
        val sigma_2_prior = 1/(rho_0 + (x.size/ sigma_2_old))
        val sigma_prior = sqrt(sigma_2_prior)
        var mu_prior = (mu_rho_0 + (x.reduce(_+_).toDense / sigma_2_old)) *
                       sigma_2_prior
        val mu_prime = DenseVectorRow.randn(mu_rho_0.length)
        mu_prime :* sigma_prior :+ mu_prior
    }
    */

    def updateComponent(theta: Theta, x: VectorRow[Double], delta: Double) =
        if (delta >= 0)
            (theta._1+delta, theta._2, theta._3)
        else
            (theta._1+delta, theta._2, theta._3)
}
