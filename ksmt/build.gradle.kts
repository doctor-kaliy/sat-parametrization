import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.7.10"
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    // core
    implementation("io.ksmt:ksmt-core:0.5.8")
    // cvc5 solver
    implementation("io.ksmt:ksmt-cvc5:0.5.8")
    // z3 solver
    implementation("io.ksmt:ksmt-z3:0.5.8")
    // yices
    implementation("io.ksmt:ksmt-yices:0.5.8")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

application {
    mainClass.set("MainKt")
}