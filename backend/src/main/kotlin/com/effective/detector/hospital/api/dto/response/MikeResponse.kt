package com.effective.detector.hospital.api.dto.response

import com.effective.detector.hospital.domain.Mike

data class MikeResponse(
    val id: Long,
    val content: String,
) {
    companion object {
        fun from(mike: Mike?): MikeResponse? {
            if (mike == null) {
                return null
            }
            return MikeResponse(
                id = mike.id!!,
                content = mike.name,
            )
        }
    }
}
