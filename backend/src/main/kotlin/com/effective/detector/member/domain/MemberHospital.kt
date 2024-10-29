package com.effective.detector.member.domain

import com.effective.detector.common.entity.BaseEntity
import com.effective.detector.hospital.domain.Hospital
import jakarta.persistence.Entity
import jakarta.persistence.FetchType
import jakarta.persistence.JoinColumn
import jakarta.persistence.ManyToOne

@Entity
class MemberHospital(
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "member_id")
    val member: Member,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "hospital_id")
    val hospital: Hospital,
) : BaseEntity() {
}
