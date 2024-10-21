package com.effective.detector.hospital.domain

import com.effective.detector.common.entity.BaseEntity
import jakarta.persistence.*

@Entity
class Slot(
    @Column(nullable = false)
    var value: Int? = null,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "hospital_id", nullable = false)
    val hospital: Hospital,

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "camera_id")  // Camera의 외래 키로 사용
    var camera: Camera? = null,
) : BaseEntity() {

    fun change(camera: Camera) {
        this.camera = camera
        camera.slot = this
    }

    fun remove() {
        camera?.slot = null
        this.camera = null
    }
}
