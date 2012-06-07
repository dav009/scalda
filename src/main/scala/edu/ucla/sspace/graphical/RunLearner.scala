package edu.ucla.sspace.graphical

import scalala.tensor.dense.DenseVector

import scala.io.Source

import java.io.PrintWriter


object RunLearner {
    def main(args:Array[String]) {
        if (args.size != 5) {
            printf("usage: RunLearner <learnerType> <data.mat> <numClusters> <outFile> <nTrials>\n")
            System.exit(1)
        }

        val data = Source.fromFile(args(1)).getLines.map( line => 
                DenseVector[Double](line.split("\\s+").map(_.toDouble)).t
        ).toList

        val nTrials = args(4).toInt
        val k = args(2).toInt
        val reportSet = Set(0, 1, 10, 100, nTrials)
        val learner = args(0) match {
            case "nb" => new gibbs.GibbsNaiveBayes(
                nTrials, 
                DenseVector[Double](Array.fill(k)(1d))t,
                DenseVector[Double](Array.fill(2)(1d)).t,
                reportSet)
            case "isgmm" => new gibbs.InfiniteSphericalGaussianMixtureModel(nTrials, 1, 2, 2, reportSet)
            case "immm" => new gibbs.InfiniteMultinomialMixtureModel(nTrials, 1, 1, reportSet)
            case "gmm" => new gibbs.FiniteGaussianMixtureModel(nTrials, 5, reportSet)
            case "km" => new gibbs.FiniteGaussianMixtureModel(nTrials, 1, reportSet, true)
            case "gdpmm" => new gibbs.DirichletProcessMixtureModel(nTrials, 1, reportSet)
            case "dpvmf" => new gibbs.InfiniteVonMisesFisherMixtureModel(nTrials, 1d, reportSet)
            case "vmf" => new em.FiniteMixtureVonMisesFisher(nTrials, reportSet)
        }

        def reporter() = data.map( x_j => "%f %f".format(x_j(0), x_j(1)) )
        val w = new PrintWriter(args(3))
        learner.setReporter(w, reporter)

        w.println("X Y Iteration Group")
        val assignments = learner.train(data.toList, k)
        w.close
    }
}
