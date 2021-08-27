package org.jetbrains.kotlinx.multik.cuda

import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.cuda.linalg.CudaLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import kotlin.concurrent.thread
import kotlin.random.Random
import kotlin.system.measureTimeMillis

private val logger = KotlinLogging.logger {}

const val NUM_WORKERS = 3

const val seed = 23
val random = Random(seed)
const val size = 300
private val mat1 = mk.d2array(size, size) { random.nextFloat() }
private val mat2 = mk.d2array(size, size) { random.nextFloat() }

class Worker1(val threaded: Boolean) : Runnable {
    private var result: NDArray<Float, D2>? = null

    override fun run() {
        if (threaded)
            CudaEngine.initCuda()

        logger.info { "Running on ${Thread.currentThread()} ${CudaEngine.getContext().handle}" }

        result = CudaLinAlg.dot(mat1, mat2)

        if (threaded)
            CudaEngine.deinitCuda()
    }

    fun result(): NDArray<Float, D2> {
        return result!!
    }
}

fun streamSol(): NDArray<Float, D2> {
    val workers = Array(NUM_WORKERS) { Worker1(true) }

    val threads = ArrayList<Thread>()
    for (worker in workers) {
        threads.add(thread { worker.run() })
    }

    for (t in threads) {
        t.join()
    }

    val result: NDArray<Float, D2> = mk.empty(size, size)

    for (worker in workers)
        result += worker.result()

    return result
}

fun naiveSol(): NDArray<Float, D2> {
    val result: NDArray<Float, D2> = mk.empty(size, size)

    val workers = Array(NUM_WORKERS) { Worker1(false) }

    for (worker in workers) {
        worker.run()
        result += worker.result()
    }

    return result
}

fun benchmark() {
    CudaEngine.initCuda()

    logger.info { "Starting naive" }
    val naive: NDArray<Float, D2>
    val naiveTime = measureTimeMillis {
        naive = naiveSol()
    } / 1000.0

    logger.info { "Naive finished. Time: $naiveTime s" }

    logger.info { "Starting stream solution" }

    val stream: NDArray<Float, D2>
    val streamTime = measureTimeMillis {
        stream = streamSol()
    } / 1000.0

    logger.info { "Stream solution finished. Time: $streamTime s" }


    assertFloatingNDArray(naive, stream)
    CudaEngine.deinitCuda()
}

fun main() {
    benchmark()
}