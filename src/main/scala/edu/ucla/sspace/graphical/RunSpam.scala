package edu.ucla.sspace.graphical

import scalala.tensor.dense.DenseVectorRow

import scala.io.Source

import java.io.PrintWriter


object RunSpam {
    def main(args:Array[String]) {
        if (args.size != 4) {
            printf("usage: RunMcDonalds <learnerType> <data.mat> <numClusters> <outFile>\n")
            System.exit(1)
        }

        val menu = Source.fromFile(args(1)).getLines.map( line => {
                val d = line.split(",")
               val v = DenseVectorRow[Double](d.view(0, d.size-4)
                                                .map(_.toDouble)
                                                .toArray)
                (d(d.size-1), v)
        }).toList

        val nTrials = 100
        val k = args(2).toInt
        val v = menu(0)._2.length
        val learner = args(0) match {
            case "nb" => new gibbs.GibbsNaiveBayes(
                nTrials, 
                new DenseVectorRow[Double](Array.fill(k)(1d)),
                new DenseVectorRow[Double](Array.fill(v)(1d)))
            case "immm" => new gibbs.InfiniteMultinomialMixtureModel(nTrials, 1, 1)
            case "gmm" => new gibbs.FiniteGaussianMixtureModel(nTrials, 1)
            case "km" => new gibbs.FiniteGaussianMixtureModel(nTrials, 1, useKMeans = true)
            case "vdpmm" => new varbayes.DirichletProcessMixtureModel(nTrials, 1)
            case "gdpmm" => new gibbs.DirichletProcessMixtureModel(nTrials, 1)
        }

        val assignments = learner.train(menu.map(_._2), k, null)
        val w = new PrintWriter(args(3))
        w.println("Spam Group")
        for ( (d, l) <- menu.zip(assignments))
            w.println("%s %d".format(d._1, l))
        w.close
    }
}
