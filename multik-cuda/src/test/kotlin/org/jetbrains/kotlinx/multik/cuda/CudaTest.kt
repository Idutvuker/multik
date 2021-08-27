package org.jetbrains.kotlinx.multik.cuda

import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.cuda.linalg.CudaLinAlg
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.slf4j.simple.SimpleLogger

val MB = 1024 * 1024L

object CudaTest {
    private val logger = KotlinLogging.logger {}

    fun testGCFreeing() {
        logger.info { "Test GC Freeing" }

        val size = 100 * MB / 4

        CudaEngine.runWithCuda {
            for (i in 0 until 10) {
                val heapFreeSize = Runtime.getRuntime().freeMemory() / MB
                val heapTotalSize = Runtime.getRuntime().totalMemory() / MB
                logger.info { "Heap size: $heapFreeSize/$heapTotalSize MB" }

                val (result, _) = CudaEngine.getContext().cache.alloc<Float, D1>(
                    size.toInt(), DataType.FloatDataType,
                    intArrayOf(size.toInt()), D1
                )

                result[1]
                superDuperGC()
                logger.info {"\n\n\n"}
            }
        }
    }

    fun testCacheFreeing() {
        logger.info { "Test Cache Freeing" }
        val heapFreeSize = Runtime.getRuntime().freeMemory() / MB
        logger.info { "heapFreeSize $heapFreeSize MB" }

        val size = MB / 4

        CudaEngine.runWithCuda {
            val arr1 = mk.empty<Float, D1>(800 * size.toInt())
            val arr2 = mk.empty<Float, D1>(800 * size.toInt())
            val arr3 = mk.empty<Float, D1>(1600 * size.toInt())
            val arr4 = mk.empty<Float, D1>(1600 * size.toInt())

            val cache = CudaEngine.getContext().cache
            cache.getOrAlloc(arr1)
            cache.getOrAlloc(arr2)
            cache.getOrAlloc(arr3)
            cache.getOrAlloc(arr4)
        }
    }

    fun `big array test`() {
        val N = 60000 / 2

        val m1 = mk.empty<Float, D2>(N, 1)
        val m2 = mk.empty<Float, D2>(1, N)

        CudaEngine.runWithCuda {
            CudaLinAlg.dot(m1, m2)
        }
    }

    fun deferredLoadTest() {
        CudaEngine.runWithCuda {
            val mat1 = mk.ndarray(
                mk[
                        mk[1.0, 2.0, 3.0],
                        mk[4.0, 5.0, 6.0],
                        mk[7.0, 8.0, 9.0],
                ]
            )

            var last = mat1

            repeat(10) {
                last = CudaLinAlg.dot(last, last)
            }

            println(last)
        }
    }

    fun deferredLoadTest2() {
        CudaEngine.runWithCuda {
            val mat1 = mk.ndarray(
                mk[
                        mk[1.0, 2.0, 3.0],
                        mk[4.0, 5.0, 6.0],
                        mk[7.0, 8.0, 9.0],
                ]
            )

            val mat2 = CudaLinAlg.dot(mat1, mat1)[1..3]

            superDuperGC() // CudaLinAlg.dot(mat1, mat1) gets GCed

            CudaEngine.getContext().cache.getOrAlloc(mat1) // this runs cleanup
            println(mat2)
        }
    }

    private fun superDuperGC() {
        repeat(5) {
            System.gc()
            Thread.sleep(50)
        }
    }

    fun abc() {
        val mat1 = mk.ndarray(
            mk[
                    mk[1.0, 2.0, 3.0],
                    mk[4.0, 5.0, 6.0],
                    mk[7.0, 8.0, 9.0],
            ]
        )

        val mat2 = CudaLinAlg.dot(mat1, mat1)[1..3]

        superDuperGC()

        CudaEngine.getContext().cache.getOrAlloc(mat1)

        println(mat2)
    }
}

fun main() {
    System.setProperty(SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "TRACE")

//    CudaTest.testCacheFreeing()
//    CudaTest.testGCFreeing()
//    benchmark()

//    CudaTest.deferredLoadTest()
//    CudaTest.deferredLoadTest2()

//    CudaTest.`big array test`()
//
//    CudaTest.testCacheFreeing()
}