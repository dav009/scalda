
name := "graphical"

organization := "edu.ucla.sspace"

version := "0.0.1"

scalaVersion := "2.9.2"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"

publishMavenStyle := true

publishTo <<= version { (v: String) =>
  val nexus = "https://oss.sonatype.org/"
  if (v.trim.endsWith("SNAPSHOT")) 
    Some("snapshots" at nexus + "content/repositories/snapshots") 
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

pomExtra := (
  <url>http://www.github.com/fozziethebeat/scalda</url>
  <licenses>
    <license>
      <name>GNU General Public License 2</name>
      <url>http://www.gnu.org/licenses/old-licenses/gpl-2.0.html</url>
      <distribution>repo</distribution>
    </license>
  </licenses>
  <scm>
    <connection>scm:git:git@github.com:fozziethebeat/scalda.git</connection>
    <developerConnection>scm:git:git@github.com:fozziethebeat/scalda.git</developerConnection>
    <url>http://github.com/fozziethebeat/scalda</url>
  </scm>
  <developers>
    <developer>
      <id>fozziethebeat</id>
      <name>Keith Stevens</name>
      <url>http://fozziethebeat.github.com</url>
    </developer>
  </developers>)

libraryDependencies ++= Seq(
    "org.scalanlp" %% "breeze-math" % "0.1-SNAPSHOT",
    "org.scalanlp" %% "breeze-learn" % "0.1-SNAPSHOT",
    "edu.ucla.sspace" % "sspace-wordsi" % "2.0",
    "com.googlecode.netlib-java" % "netlib-java" % "0.9.3",
    "colt" % "colt" % "1.2.0",
    "org.apache.commons" % "commons-math3" % "3.0"
)

scalacOptions += "-deprecation"
