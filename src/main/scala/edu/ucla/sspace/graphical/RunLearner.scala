package edu.ucla.sspace.learner

import edu.ucla.sspace.learner.gibbs._

import scalala.library.Library._
import scalala.collection.sparse.SparseArray
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.sparse.SparseVectorRow

import scala.io.Source

import java.io.PrintWriter


object RunLearner {
    def main(args:Array[String]) {
        val data = Source.fromFile(args(1)).getLines.map( line => {
                val d = line.split("\\s+").map(_.toDouble)
                val v = new SparseVectorRow[Double](
                    new SparseArray[Double](d.size, (0 until d.size).toArray, d, d.size, d.size))
                v
        }).toList

        val nTrials = 100
        val k = args(2).toInt
        val learner = args(0) match {
            case "nb" => new GibbsNaiveBayes(
                nTrials, 
                new DenseVectorRow[Double](Array.fill(k)(1d)),
                new DenseVectorRow[Double](Array.fill(2)(1d)))
            case "gmm" => new FiniteGaussianMixtureModel(nTrials, 5)
            case "dpmm" => new DirichletProcessMixtureModel(nTrials, 1)
        }

        val assignments = learner.train(data, k)
        val w = new PrintWriter(args(3))
        for ( (d, l) <- data.zip(assignments))
            w.println("%f %f %d".format(d(0), d(1), l))
        w.close
    }
}
