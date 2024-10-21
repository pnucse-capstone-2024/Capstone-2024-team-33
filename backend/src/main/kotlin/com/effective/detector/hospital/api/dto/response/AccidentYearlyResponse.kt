package com.effective.detector.hospital.api.dto.response

data class AccidentYearlyResponse(
    val year: Int,
    val slipping: Int,
    val fighting: Int,
    val poorFacilities: Int,
    val carelessness: Int,
    val malfunction: Int,
    val etc: Int,
    val total: Int,
)
