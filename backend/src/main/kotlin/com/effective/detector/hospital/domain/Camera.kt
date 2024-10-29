package com.effective.detector.hospital.domain

import com.effective.detector.common.entity.BaseEntity
import jakarta.persistence.*

@Entity
class Camera(
    @Column(nullable = false)
    val name: String,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "hospital_id", nullable = false)
    val hospital: Hospital,

    @OneToOne(mappedBy = "camera", fetch = FetchType.LAZY)
    var slot: Slot? = null,

    @OneToMany(mappedBy = "camera", cascade = [CascadeType.ALL], fetch = FetchType.LAZY)
    val accidents: List<Accident> = mutableListOf(),
) : BaseEntity()
