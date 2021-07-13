/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_jvm.linAlg


import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.jvm.JvmLinAlg
import org.jetbrains.kotlinx.multik.jvm.JvmLinAlg.dot
import org.jetbrains.kotlinx.multik.jvm.JvmLinAlg.inv
import org.jetbrains.kotlinx.multik.jvm.JvmLinAlg.solve
import org.jetbrains.kotlinx.multik.jvm.JvmMath.max
import org.jetbrains.kotlinx.multik.jvm.JvmMath.min
import org.jetbrains.kotlinx.multik.jvm.PLU
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.math.abs
import kotlin.math.max

import kotlin.random.Random
import kotlin.system.measureTimeMillis
import kotlin.test.Test
import kotlin.test.assertEquals

class JvmLinAlgTest {

    @Test
    fun `test of norm function with p=1`() {
        val d2arrayDouble1 = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val d2arrayDouble2 = mk.ndarray(mk[mk[-1.0, -2.0], mk[-3.0, -4.0]])

        assertEquals(10.0, mk.linalg.norm(d2arrayDouble1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayDouble2, 1))

        val d2arrayShort1 = mk.ndarray(mk[mk[1.toShort(), 2.toShort()], mk[3.toShort(), 4.toShort()]])
        val d2arrayShort2 = mk.ndarray(mk[mk[(-1).toShort(), (-2).toShort()], mk[(-3).toShort(), (-4).toShort()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayShort1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayShort2, 1))

        val d2arrayInt1 = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val d2arrayInt2 = mk.ndarray(mk[mk[-1, -2], mk[-3, -4]])

        assertEquals(10.0, mk.linalg.norm(d2arrayInt1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayInt2, 1))

        val d2arrayByte1 = mk.ndarray(mk[mk[1.toByte(), 2.toByte()], mk[3.toByte(), 4.toByte()]])
        val d2arrayByte2 = mk.ndarray(mk[mk[(-1).toByte(), (-2).toByte()], mk[(-3).toByte(), (-4).toByte()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayByte1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayByte2, 1))

        val d2arrayFloat1 = mk.ndarray(mk[mk[1.0.toFloat(), 2.0.toFloat()], mk[3.0.toFloat(), 4.0.toFloat()]])
        val d2arrayFloat2 = mk.ndarray(mk[mk[(-1.0).toFloat(), (-2.0).toFloat()], mk[(-3.0).toFloat(), (-4.0).toFloat()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayFloat1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayFloat2, 1))

        val d2arrayLong1 = mk.ndarray(mk[mk[1.toLong(), 2.toLong()], mk[3.toLong(), 4.toLong()]])
        val d2arrayLong2 = mk.ndarray(mk[mk[(-1).toLong(), (-2).toLong()], mk[(-3).toLong(), (-4).toLong()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayLong1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayLong2, 1))

    }

//    @Test
//    fun `test solveTriangleInplace`() {
//        val procedurePrecision = 1e-5
//
//        val rnd = Random(123)
//        val n = 18
//        val m = 100
//
//        val a = mk.d2array<Double>(n, n) {0.0}
//        for (i in 0 until n) {
//            for (j in 0 .. i) {
//                if (j == i) {
//                    a[i, j] = 1.0
//                } else {
//                    a[i, j] = rnd.nextDouble()
//                }
//            }
//        }
//
//        val b = mk.d2array<Double>(n, m) {0.0}
//        for (i in 0 until n)
//            for (j in 0 until m)
//                b[i, j] = rnd.nextDouble()
//
//        val saveB = b.deepCopy()
//
//        JvmLinAlg.solveTriangleInplace(a, 0, 0, n, b, 0, 0, m)
//
//        //check equality up to precision
//        assert(max(dot(a, b) - saveB) < procedurePrecision && min(dot(a, b) - saveB) > -procedurePrecision)
//    }

//    @Test
//    fun `gemm test`() {
//        val procedurePrecision = 1e-5
//
//        val a = mk.ndarray(mk[mk[1.0, 2.0, 3.0, 2.0, 1.0], mk[4.0, 5.0, 6.0, 5.0, 4.0]])
//        val b = mk.ndarray(mk[mk[7.0, 8.0], mk[9.0, 10.0], mk[11.0, 12.0], mk[13.0, 14.0], mk[15.0, 16.0]])
//        val c = mk.ndarray(mk[mk[11.0, 13.0], mk[15.0, 17.0]])
//        val adotb = mk.ndarray(mk[mk[110.0, 121.0], mk[279.0, 305.0]])
//
//        //        proof: numpy
//        //
//        //        a = np.array([[1.0, 2.0, 3.0, 2.0, 1.0], [4.0, 5.0, 6.0, 5.0, 4.0]])
//        //        b = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
//        //        print(a @ b + c)
//
//        gemm(a, 0, 0, 2, 5, 1.0,
//             b, 0, 0, 5, 2,
//             c, 0, 0, 2, 2, 1.0)
//
//        assert(c == adotb)
//
//    }



    @Test
    fun `test plu`() {

        val procedurePrecision = 1e-5

        val iters = 10000
        val sideFrom = 1
        val sideUntil = 100
        for (all in 0 until iters) {

            val rnd = Random(System.currentTimeMillis())
            val m = rnd.nextInt(sideFrom, sideUntil)
            val n = rnd.nextInt(sideFrom, sideUntil)


            val a = mk.d2array<Double>(m, n) { rnd.nextDouble() }

            val (P, L, U) = PLU(a)

            val diff = a - dot(P, dot(L, U))
            val maxdiff = max(diff)
            val mindiff = min(diff)

            val ok = maxdiff < procedurePrecision && maxdiff - mindiff < procedurePrecision


            if(!ok) {
                println("wrong on:")
                println(a)
                println("PLU gives:")
                println(P)
                println(L)
                println(U)
                println("P*L*U = ")
                println(dot(P, dot(L, U)))
                assert(false)

            }

        }
    }

    @Test
    fun `plu benchmark`() {
        val rnd = Random(424242)
        val iters = 1000
        val maxSideLen = 400
        var avgTime = 0.0
        var maxTime = 0.0
        var minTime = Double.POSITIVE_INFINITY


        for (all in 0 until iters) {

            val m = rnd.nextInt(maxSideLen / 2, maxSideLen + 1)
            val n = rnd.nextInt(maxSideLen / 2, maxSideLen + 1)


            val a = mk.d2array<Double>(m, n) { rnd.nextDouble() }

            val newtime = measureTimeMillis {
                val (P, L, U) = PLU(a)
            }.toDouble()
            maxTime = kotlin.math.max(maxTime, newtime)
            minTime = kotlin.math.min(minTime, newtime)
            avgTime += newtime
        }
        avgTime /= iters
        println("On $iters iteration on random matrices with random side in range [${maxSideLen / 2}, $maxSideLen]:\n" +
                "avgTime: $avgTime ms\nmaxTime: $maxTime ms\nminTime: $minTime ms")
    }


    @Test
    fun `solve test`() {
        val procedurePrecision = 1e-5
        val rnd = Random(System.currentTimeMillis())
        for (iteration in 0 until 1000) {
            //test when second argument is d2 array
            val maxlen = 100
            val n = rnd.nextInt(1, maxlen)
            val m = rnd.nextInt(1, maxlen)
            val a = mk.d2array<Double>(n, n) { rnd.nextDouble() }
            val b = mk.d2array<Double>(n, m) { rnd.nextDouble() }
            var solDelta = dot(a, solve(a, b)) - b
            var solDeltamax = max(solDelta)
            var solDeltamin = min(solDelta)
            assert(abs(solDeltamax) < procedurePrecision && abs(solDeltamin) < procedurePrecision)

            //test when second argument is d1 vector
            val bd1 = mk.d1array(n) { rnd.nextDouble() }
            val bd1Transpose = mk.d2array(bd1.size, 1) { 0.0 }
            for (i in 0 until bd1.size) {
                bd1Transpose[i, 0] = bd1[i]
            }

            val sol = solve(a, bd1)
            val solTranspose = mk.d2array<Double>(sol.size, 1) { 0.0 }
            for (i in 0 until sol.size) {
                solTranspose[i, 0] = sol[i]
            }

            solDelta = dot(a, solTranspose) - bd1Transpose
            solDeltamax = max(solDelta)
            solDeltamin = min(solDelta)
            assert(abs(solDeltamax) < procedurePrecision && abs(solDeltamin) < procedurePrecision)


        }
    }

    @Test
    fun whatever() {
        val b = mk.ndarray(mk[
                mk[1.5, 2.1, 3.0],
                mk[4.0, 5.0, 6.0],
                mk[7.0, 8.0, 9.0]
            ])
        val bslice = b[1..2, 1..2] as D2Array<Double>
        println(bslice)
        bslice[0, 0] = .0
        println(b)

    //        val a = mk.ndarray(mk[mk[-1.0, 2.0, 3.0], mk[4.0, 5.0, 6.0], mk[7.0, 8.0, 10.0]])
//        val b = mk.ndarray(mk[mk[7.0], mk[6.0], mk[5.0]])
//        println(dot(a, solve(a, b)))
//        println(inv(a))

    }
}