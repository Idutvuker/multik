package org.jetbrains.multik.api

import java.util.*
import java.util.concurrent.ConcurrentHashMap

public enum class EngineType(name: String) {
    JVM("jvm"), NATIVE("native")
}


public abstract class Engine {

    protected abstract val name: String

    protected abstract val type: EngineType

    protected val engines: MutableMap<EngineType, Engine> = ConcurrentHashMap<EngineType, Engine>()

    protected var defaultEngine: EngineType? = null

    protected fun loadEngine() {
        val loaders: ServiceLoader<EngineProvider> = ServiceLoader.load(EngineProvider::class.java)
        for (engineProvider in loaders) {
            val engine = engineProvider.getEngine()
            if (engine != null) {
                engines[engine.type] = engine
            }
        }

        if (engines.isEmpty()) {
            throw EngineMultikException("The map of engines is empty.")
        }

        defaultEngine = if (engines.containsKey(EngineType.JVM)) {
            EngineType.JVM
        } else {
            engines.iterator().next().key
        }
    }


    public abstract fun getMath(): Math
    public abstract fun getLinAlg(): LinAlg

    internal companion object : Engine() {

        init {
            loadEngine()
        }

        override val name: String
            get() = throw EngineMultikException("For a companion object, the name is undefined.")

        override val type: EngineType
            get() = throw EngineMultikException("For a companion object, the type is undefined.")

        internal fun getDefaultEngine(): String? = defaultEngine?.name

        internal fun setDefaultEngine(type: EngineType) {
            if (!engines.containsKey(type)) throw EngineMultikException("This type of engine is not available.")
            defaultEngine = type
        }

        override fun getMath(): Math {
            if (engines.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide Math implementation.")
            return engines[defaultEngine]?.getMath()
                ?: throw EngineMultikException("The used engine type is not defined.")
        }

        override fun getLinAlg(): LinAlg {
            if (engines.isEmpty()) throw EngineMultikException("The map of engines is empty. Can not provide LinAlg implementation.")
            return engines[defaultEngine]?.getLinAlg()
                ?: throw throw EngineMultikException("The used engine type is not defined.")
        }
    }
}

public interface EngineProvider {
    public fun getEngine(): Engine?
}

public class EngineMultikException(message: String) : Exception(message) {
    public constructor() : this("")
}