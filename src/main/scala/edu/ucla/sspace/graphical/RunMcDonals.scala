package edu.ucla.sspace.graphical

import edu.ucla.sspace.graphical.gibbs.{GibbsNaiveBayes,FiniteGaussianMixtureModel}
import edu.ucla.sspace.graphical.varbayes.DirichletProcessMixtureModel

import scalala.tensor.dense.DenseVectorRow

import scala.io.Source

import java.io.PrintWriter


object RunMcDonalds {
    def main(args:Array[String]) {
        if (args.size != 4) {
            printf("usage: RunMcDonalds <learnerType> <data.mat> <numClusters> <outFile>\n")
            System.exit(1)
        }

        val menu = Source.fromFile(args(1)).getLines.map( line => {
                val d = line.split("\\|")
                val c = d(2).toDouble + 0.00000011
                val v = new DenseVectorRow[Double](d.view(3, d.size).map(v => if (v == "N/A") 0d else v.toDouble/c).toArray)
                (d(0).replaceAll("\\s+", "_"), d(1).replaceAll("\\s+", "_"), v)
        }).toList

        val nTrials = 100
        val k = args(2).toInt
        val v = menu(0)._3.length
        val learner = args(0) match {
            case "nb" => new GibbsNaiveBayes(
                nTrials, 
                new DenseVectorRow[Double](Array.fill(k)(1d)),
                new DenseVectorRow[Double](Array.fill(v)(1d)))
            case "gmm" => new FiniteGaussianMixtureModel(nTrials, 1)
            case "km" => new FiniteGaussianMixtureModel(nTrials, 1, true)
            case "vdpmm" => new varbayes.DirichletProcessMixtureModel(nTrials, 1)
            case "gdpmm" => new gibbs.DirichletProcessMixtureModel(nTrials, 1)
        }

        val assignments = learner.train(menu.map(_._3), k)
        val w = new PrintWriter(args(3))
        w.println("Category Item Group")
        for ( (d, l) <- menu.zip(assignments))
            w.println("%s %s %d".format(d._1, d._2, l))
        w.close
    }
}
