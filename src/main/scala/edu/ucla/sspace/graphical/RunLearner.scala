package edu.ucla.sspace.graphical

import DistanceMetrics.euclidean

import breeze.linalg.DenseVector

import scala.io.Source
import scala.math.pow

import java.io.PrintWriter


object RunLearner {
    def main(args:Array[String]) {
        if (args.size != 5) {
            printf("usage: RunLearner <learnerType> <data.mat> <numClusters> <outFile> <nTrials>\n")
            System.exit(1)
        }

        val data = Source.fromFile(args(1)).getLines.map( line => 
                DenseVector[Double](line.split("\\s+").map(_.toDouble))
        ).toList

        val nTrials = args(4).toInt
        val k = args(2).toInt
        val v = data(0).length
        val reportSet = Set(0, 1, 10, 100, nTrials)
        val learner = args(0) match {
            /*
            case "nb" => new gibbs.GibbsNaiveBayes(
                nTrials, 
                DenseVector[Double](Array.fill(k)(1d))t,
                DenseVector[Double](Array.fill(2)(1d)).t,
                reportSet)
            */

            case "isgmm" => new gibbs.InfiniteSphericalGaussianMixtureModel(nTrials, 1, getGenerator(data), reportSet)
            /*
            case "immm" => new gibbs.InfiniteMultinomialMixtureModel(nTrials, 1, 1, reportSet)
            case "gdpmm" => new gibbs.DirichletProcessMixtureModel(nTrials, 1, reportSet)
            case "dpvmf" => new gibbs.InfiniteVonMisesFisherMixtureModel(nTrials, 1d, reportSet)

            case "sgmm" => new gibbs.FiniteSphericalGaussianMixtureModel(nTrials, 1, getGenerator(data), reportSet)
            case "gmm" => new gibbs.FiniteGaussianMixtureModel(nTrials, 5, reportSet)
            case "km" => new gibbs.FiniteGaussianMixtureModel(nTrials, 1, reportSet, true)
            case "vmf" => new em.FiniteMixtureVonMisesFisher(nTrials, reportSet)
            */

        }

        def reporter() = data.map( x_j => "%f %f".format(x_j(0), x_j(1)) )
        val w = new PrintWriter(args(3))
        learner.setReporter(w, reporter)

        w.println("X Y Iteration Group")
        val assignments = learner.train(data.toList, k, null)
        w.close
    }

    def getGenerator(data: List[DenseVector[Double]]) = {
        val mu = data.reduce(_+_) / data.size.toDouble
        val variance = data.map(euclidean(_,mu)).map(pow(_,2)).sum / data.size.toDouble
        new SphericalGaussianRasmussen(mu,  variance)
        //new SphericalGaussianMaximumLikelihood()
    }
}
