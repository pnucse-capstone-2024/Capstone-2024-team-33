package com.effective.detector.hospital.api.dto.response

import com.effective.detector.hospital.domain.Accident
import com.effective.detector.hospital.domain.AccidentType
import com.effective.detector.hospital.domain.AgeType
import java.time.LocalDate
import java.time.LocalDateTime

data class AccidentResponse(
    val id: Long,
    val videoUrl: String,
    val startTime: LocalDateTime,
    val endTime: LocalDateTime,
    val date: LocalDate,
    val type: TypeResponse,
    val age: AgeResponse,
    val camera: CameraResponse?,
    val mike: MikeResponse?,
) {
    companion object {
        fun from(accident: Accident): AccidentResponse {
            return AccidentResponse(
                id = accident.id!!,
                videoUrl = accident.videoUrl ?: "",
                startTime = accident.startTime,
                endTime = accident.endTime,
                date = accident.startTime.toLocalDate(),
                type = TypeResponse(
                    id = accident.type!!.id,
                    content = accident.type!!.value,
                ),
                age = AgeResponse(
                    id = accident.age!!.id,
                    content = accident.age!!.value,
                ),
                camera = CameraResponse.from(accident.camera),
                mike = MikeResponse.from(accident.mike),
            )
        }
    }
}

data class TypeResponse(
    val id: Long,
    val content: String,
) {
    companion object {
        fun from(type: AccidentType): TypeResponse {
            return TypeResponse(
                id = type.id,
                content = type.value,
            )
        }
    }
}

data class AgeResponse(
    val id: Long,
    val content: String,
) {
    companion object {
        fun from(type: AgeType): AgeResponse {
            return AgeResponse(
                id = type.id,
                content = type.value,
            )
        }
    }
}
