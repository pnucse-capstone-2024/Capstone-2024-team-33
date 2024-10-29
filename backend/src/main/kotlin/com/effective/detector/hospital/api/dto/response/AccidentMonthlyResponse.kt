package com.effective.detector.hospital.api.dto.response

data class AccidentMonthlyResponse(
    val month: String,
    val slipping: Int,
    val fighting: Int,
    val poorFacilities: Int,
    val carelessness: Int,
    val malfunction: Int,
    val etc: Int,
    val total: Int,
)
