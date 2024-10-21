package com.effective.detector.common.util

fun Any.logInfo(message: String) {
    val logger = org.slf4j.LoggerFactory.getLogger(this.javaClass)
    logger.info(message)
}

fun Any.logWarn(message: String? = null, e: Exception) {
    val logger = org.slf4j.LoggerFactory.getLogger(this.javaClass)
    logger.warn(message ?: e.message ?: e.localizedMessage, e)
}

fun Any.logError(message: String? = null, e: Exception) {
    val logger = org.slf4j.LoggerFactory.getLogger(this.javaClass)
    logger.error(message ?: e.message ?: e.localizedMessage, e)
}
