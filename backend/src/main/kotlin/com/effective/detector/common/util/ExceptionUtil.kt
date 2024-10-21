package com.effective.detector.common.util

import com.effective.detector.common.error.BusinessError
import com.effective.detector.common.error.BusinessException
import org.springframework.data.repository.CrudRepository
import org.springframework.data.repository.findByIdOrNull

fun <T, ID> CrudRepository<T, ID>.findByIdOrThrow(
    id: ID,
): T {
    return this.findByIdOrNull(id) ?: throw BusinessException(
        BusinessError.ID_NOT_FOUND
    )
}
