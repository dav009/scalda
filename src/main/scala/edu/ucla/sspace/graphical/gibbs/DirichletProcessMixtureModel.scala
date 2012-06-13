package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Likelihood.gaussian
import edu.ucla.sspace.graphical.Util.{norm,epsilon}

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Gamma
import scalanlp.stats.distributions.Multinomial


class DirichletProcessMixtureModel(val numIterations: Int, 
                                   val alpha: Double,
                                   s: Set[Int] = Set[Int]()) extends Learner {
    type Theta = (Double, DenseVectorRow[Double], DenseVectorRow[Double])

    val alpha_vec = DenseVectorRow(alpha)
    val beta = 1d
    val gamma = 1d

    def train(data: List[VectorRow[Double]],
              ignored: Int,
              ignored2: List[List[VectorRow[Double]]]) = {
        // Extract the shape of the data.
        val n = data.size
        val v = data(0).length
        val t = n - 1 + alpha

        // Compute the global mean of all data points.
        val mu_0 = data.reduce(_+_).toDense / n
        // Compute the variance of all data points to the global mean.
        val variance_0 = (data.map(x_j => (mu_0 - x_j) :^ 2).reduce(_+_) / n) + epsilon
        val rho_0 = variance_0.map(1/_)
        val sigma_prime = DenseVectorRow.ones[Double](v)

        def newTheta(x:List[VectorRow[Double]],
                     sigma_old: DenseVectorRow[Double] = sigma_prime) =
             sampleTheta(mu_0 :* rho_0, rho_0, sigma_old, x)

        // Create the global component that will be used to determine when a 
        // new component should be sampled.
        //val components = Array((alpha, mu_0, variance_0)).toBuffer
        val components = Array((alpha, mu_0, variance_0)).toBuffer

        // Setup the initial labels for all the data points.  These start off 
        // with no meaningful value.
        var labels = Array.fill(n)(0)

        if (s contains 0)
            report(0, labels.toList)

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
                val oldComponent:Theta = if (i != 0) {
                    val (n_lj, mu, sigma) = components(l_j)
                    components(l_j) = (n_lj-1, mu, sigma)
                    (n_lj, mu, sigma)
                } else null

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
                    components.append(newTheta(List(x_j))) //newComponent(x_j))
                } else {
                    // Restore the bookeeping information for this point using the
                    // old assignment.
                    labels(j) = l_j_new
                    components(l_j_new) = updateComponent(components(l_j_new))
                }
            }

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
                components(c+1) = newTheta(x.toList, sigmas(c_old))
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

    def sampleTheta(mu_rho_0: DenseVectorRow[Double],
                    rho_0: DenseVectorRow[Double],
                    sigma_old: DenseVectorRow[Double],
                    x:List[VectorRow[Double]]) = {
        val sigma_sqr_prior = (rho_0 :+ sigma_old.mapValues(x.size/_)).map(1/_)
        val sigma_prior = sigma_sqr_prior :^ 0.5
        var mu_prior = mu_rho_0
        mu_prior += x.reduce(_+_).toDense :/ sigma_old
        mu_prior = mu_prior :* sigma_sqr_prior

        val mu_prime = DenseVectorRow.randn(mu_rho_0.length)
        val mu = mu_prime :* sigma_prior :+ mu_prior

        val beta_prior = beta //+ x.size / 2d
        val gamma_prior = x.map(_:-mu).map(_:^2).reduce(_+_).toDense/x.size + gamma
        val rho_nlp = gamma_prior.mapValues(gp=> (new Gamma(beta_prior, gp).sample))
        //println(rho_apc)
        val sigma = rho_nlp //DenseVectorRow.ones[Double](sigma_old.length) //gamma_prior.mapValues(gp=> (new Gamma(beta_prior, gp).sample))

        (x.size.toDouble, mu, sigma)
    }

    def newTheta2(x:List[VectorRow[Double]],
                 sigma_old: DenseVectorRow[Double] = null) = {
        val n_c = x.size
        val mu_c = x.reduce(_+_).toDense / n_c
        val sigma_c = (x.map(x_j => (mu_c - x_j) :^2).reduce(_+_)/n_c) + epsilon
        (n_c.toDouble, mu_c, sigma_c)
    }

    def newComponent(x: VectorRow[Double]) : Theta = 
        (1d, x.toDense, DenseVectorRow.ones[Double](x.length))
    def updateComponent(theta: Theta) =
        (theta._1+1, theta._2, theta._3)
}
