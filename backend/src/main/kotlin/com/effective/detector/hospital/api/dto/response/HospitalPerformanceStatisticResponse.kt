package com.effective.detector.hospital.api.dto.response

data class HospitalPerformanceStatisticResponse(
    val totalAccidentCount: Long,
    val primaryReason: PrimaryReasonResponse,
    val increaseRateByLastYear: Double,
    val detectionAccuracy: Double,
    val mostAccidentsOrccuredMonth: Int,
    val increaseRateByAverage: Double,
)
