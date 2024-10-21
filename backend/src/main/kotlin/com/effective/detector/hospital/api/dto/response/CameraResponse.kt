package com.effective.detector.hospital.api.dto.response

import com.effective.detector.hospital.domain.Camera

data class CameraResponse(
    val id: Long,
    val content: String,
) {
    companion object {
        fun from(camera: Camera?): CameraResponse? {
            if (camera == null) {
                return null
            }
            return CameraResponse(
                id = camera.id!!,
                content = camera.name,
            )
        }

        fun toDto(camera: Camera): CameraResponse {
            return CameraResponse(
                id = camera.id!!,
                content = camera.name,
            )
        }
    }
}
