package com.effective.detector.common.error

class BusinessException(
    val error: BusinessError,
) : RuntimeException() {
}
