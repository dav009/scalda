
name := "Scalda"

version := "0.0.1"

scalaVersion := "2.9.1"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"

libraryDependencies ++= Seq(
    "org.scalanlp" %% "breeze-math" % "0.1-SNAPSHOT",
    "org.scalanlp" %% "breeze-learn" % "0.1-SNAPSHOT",
    "edu.ucla.sspace" % "sspace-wordsi" % "2.0",
    "com.googlecode.netlib-java" % "netlib-java" % "0.9.3"
)

scalacOptions += "-deprecation"
