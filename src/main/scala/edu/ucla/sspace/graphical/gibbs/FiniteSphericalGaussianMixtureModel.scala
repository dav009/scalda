package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.ComponentGenerator
import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Likelihood._
import edu.ucla.sspace.graphical.Util.{norm,epsilon}

import breeze.numerics._
import breeze.linalg.DenseVector
import breeze.linalg.Vector

import breeze.stats.distributions.Gamma
import breeze.stats.distributions.Multinomial

import scala.util.Random


class FiniteSphericalGaussianMixtureModel(val numIterations: Int, 
                                          val alpha: Double,
                                          val generator: ComponentGenerator,
                                          s: Set[Int] = Set[Int]()) extends Learner {

    type Theta = (Double, DenseVectorRow[Double], Double)

    def train(data: List[VectorRow[Double]], k: Int) = {
        // Extract the shape of the data.
        val n = data.size
        val alpha_k = alpha/k.toDouble

        val labels = Array.fill(n)(0)
        var components = Array.fill(k)(generator.initial)

        for ((x_j,j) <- data.zipWithIndex) {
            def dataLikelihood(theta: Theta) = gaussian(x_j, theta._2, theta._3)
            val likelihood = DenseVectorRow[Double](components.map(dataLikelihood))
            val l_j = likelihood.argmax
            components(l_j) = updateComponent(components(l_j), x_j, +1)
            labels(j) = l_j
        }
        if (s contains 0)
            report(0, labels.toList)

        components = updateComponents(components, labels, data)

        val mu_k = components.map(_._2)
        val variance_k = components.map(_._3)
        generator.update(mu_k, variance_k)

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

            components = updateComponents(components, labels, data)

            val mu_k = components.map(_._2)
            val variance_k = components.map(_._3)
            generator.update(mu_k, variance_k)

            if (s contains (i+1))
                report(i+1, labels.toList)
        }

        labels
    }

    def updateComponents(components: Array[Theta],
                         labels: Array[Int],
                         data: List[VectorRow[Double]]) = {
        val groups = labels.zip(data).groupBy(_._1)
                           .map{ case(k,v) => (k, v.map(_._2).toList) }
        components.zipWithIndex.map{ case (theta, c) => 
                groups.get(c) match {
                    case Some(x) => generator.sample(x, theta._3)
                    case None => generator.initial
                }
        }
    }

    def updateComponent(theta: Theta, x: VectorRow[Double], delta: Double) =
        if (delta >= 0)
            (theta._1+delta, theta._2, theta._3)
        else
            (theta._1+delta, theta._2, theta._3)
}
