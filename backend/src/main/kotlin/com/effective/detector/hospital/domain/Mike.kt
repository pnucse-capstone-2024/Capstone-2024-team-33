package com.effective.detector.hospital.domain

import com.effective.detector.common.entity.BaseEntity
import jakarta.persistence.*

@Entity
class Mike(
    @Column(nullable = false)
    val name: String,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "hospital_id", nullable = false)
    val hospital: Hospital,

    @OneToMany(mappedBy = "mike", cascade = [CascadeType.ALL], fetch = FetchType.LAZY)
    val accidents: List<Accident> = mutableListOf(),
) : BaseEntity()
