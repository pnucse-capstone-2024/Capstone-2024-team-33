package com.effective.detector.hospital.api.dto.request

data class MonitorChangeRequest(
    val slot: Int,
    val cameraId: Long,
)
