package com.effective.detector.hospital.api.dto.response

import com.effective.detector.hospital.domain.AccidentType

data class PrimaryReasonResponse(
    val id: Long,
    val content: String,
) {
    constructor(accidentType: AccidentType) : this(
        id = accidentType.id,
        content = accidentType.value
    )
}
