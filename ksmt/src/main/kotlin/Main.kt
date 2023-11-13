import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3SMTLibParser
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.yices.KYicesSolver
import java.nio.file.FileVisitResult
import java.nio.file.Path
import kotlin.io.path.*
import kotlin.time.Duration.Companion.seconds

data class Result(val time: Long, val status: KSolverStatus)

fun runSolver(formula: Path): Result {
//    val ctx = KContext()
    // read file to string
    KContext().use { ctx ->
        // time for parsing
        val startTimeForReading = System.currentTimeMillis()

        val startTimeForParsing = System.currentTimeMillis()

        val assertion = KZ3SMTLibParser(ctx).parse(formula)

        val endTimeForParsing = System.currentTimeMillis()

        println("Time for reading: ${startTimeForParsing - startTimeForReading} ms")
        println("Time for parsing: ${endTimeForParsing - startTimeForParsing} ms")


        KCvc5Solver(ctx).use { solver -> // create a Z3 SMT solver instance
            // assert expression

            // time to Add
            val startTimeForAdding = System.currentTimeMillis()
            assertion.forEach { solver.assert(it) }
            val endTimeForAdding = System.currentTimeMillis()
            println("Time for adding: ${endTimeForAdding - startTimeForAdding} ms")

            // start time measurement
            val start = System.currentTimeMillis()

            // check assertions satisfiability with timeout
            val satisfiability = solver.check(timeout = 3600.seconds)
            println(satisfiability) // SAT

            // end time measurement
            val time = System.currentTimeMillis() - start
            println("Time: $time ms")

            return Result(time, satisfiability)
        }
    }
}

@OptIn(ExperimentalPathApi::class)
fun collectBenches() {
    val benches = Path.of("benches")
    val classes = listOf(100L, 500L, 1000L, 10000L, 50000L, 100000L)

//    benches.createDirectory()

//    classes.forEach {
//        benches.resolve(it.toString()).createDirectory()
//    }

    val benchmarkTime = benches.resolve("time.csv")

//    benchmarkTime.createFile()
    val complete = benchmarkTime.readLines().map { it.split(",")[0] }.toSet()
    val ignore = benches.resolve("ignore")
    val ignored = ignore.readLines().toSet()

    val src = Path.of("ksmt/ksmt-test/build/smtLibBenchmark/QF_ABV/QF_ABV")

    src.visitFileTree {
        onVisitFile { file, _ ->
            val relative = file.relativeTo(src)
            println(relative)
            if (!complete.contains(relative.toString()) && !ignored.contains(relative.toString())) {
                ignore.appendLines(listOf(relative.toString()))

                val (time, status) = runSolver(file)

                val cl = classes.find { time < it }
                val to = cl?.toString()?.let { benches.resolve(it).resolve(relative) }

                println(to)
                println(file)

                to?.parent?.createDirectories()
                to?.let(file::copyTo)

                benchmarkTime.appendLines(listOf("$relative,$time,$status"))
            }
            FileVisitResult.CONTINUE
        }
    }
}

fun main() {
    collectBenches()
}
