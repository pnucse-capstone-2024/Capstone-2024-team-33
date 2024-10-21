package com.effective.detector.hospital.domain

import org.springframework.data.jpa.repository.JpaRepository

interface CameraRepository : JpaRepository<Camera, Long> {
}
