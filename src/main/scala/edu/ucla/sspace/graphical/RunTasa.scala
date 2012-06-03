package edu.ucla.sspace.graphical

import edu.ucla.sspace.graphical.gibbs.{GibbsNaiveBayes,FiniteGaussianMixtureModel}
import edu.ucla.sspace.matrix.MatrixIO
import edu.ucla.sspace.matrix.MatrixIO.Format

import scalala.tensor.dense.DenseVector
import scalala.tensor.sparse.SparseVector

import scala.io.Source

import java.io.PrintWriter


object RunTasa {
    def main(args:Array[String]) {
        if (args.size != 4) {
            printf("usage: RunLearner <learnerType> <data.mat> <numClusters> <outFile>\n")
            System.exit(1)
        }

        val m = MatrixIO.readSparseMatrix(args(1), Format.MATLAB_SPARSE)
        val data = (0 until m.rows).map( r => {
            val v = m.getRowVector(r)
            val svr = SparseVector.create[Double](v.length)().t
            for (c <- v.getNonZeroIndices)
                svr(c) = v.get(c)
            svr
        }).toList

        val nTrials = 100
        val k = args(2).toInt
        val learner = args(0) match {
            case "nb" => new GibbsNaiveBayes(
                nTrials, 
                DenseVector[Double](Array.fill(k)(1d)).t,
                DenseVector[Double](Array.fill(2)(1d)).t)
            case "gmm" => new FiniteGaussianMixtureModel(nTrials, 5)
            case "km" => new FiniteGaussianMixtureModel(nTrials, 1, useKMeans=true)
            case "vdpmm" => new varbayes.DirichletProcessMixtureModel(nTrials, 1)
            case "gdpmm" => new gibbs.DirichletProcessMixtureModel(nTrials, 1)
        }

        val assignments = learner.train(data.toList, k)
        val w = new PrintWriter(args(3))
        w.println("X Y Group")
        for ( (d, l) <- data.zip(assignments))
            w.println("%d".format(l))
        w.close
    }
}
