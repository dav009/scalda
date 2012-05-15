package edu.ucla.sspace.graphical

import scala.xml._

import java.io.PrintWriter


object FormatMCDonaldsMenu {
    def main(args: Array[String]) {
        val rawMenu = XML.load(args(0))
        val pw = new PrintWriter(args(1))
        for (menuCategory <- rawMenu \\ "menucategory") {
            val categoryName = (menuCategory \ "@name").text
            for (item <- menuCategory \\ "item") {
                val itemName = (item \ "@name").text
                var featureList = List(categoryName, itemName)
                for (nutrient <- item \\ "nutrient") {
                    val nutrientName = (nutrient \ "@name").text
                    val value = ((nutrient \ "fact")(0) \ "@value").text
                    featureList = featureList :+ value
                }
                pw.println(featureList.mkString("|"))
            }
        }
        pw.close
    }
}
