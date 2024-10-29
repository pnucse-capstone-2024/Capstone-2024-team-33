package com.effective.detector.member.domain

import com.effective.detector.common.entity.BaseEntity
import com.effective.detector.hospital.domain.Hospital
import jakarta.persistence.*

@Entity
class Member(
    @Column(unique = true)
    var loginId: String? = null,

    @Column(nullable = false)
    var loginPassword: String? = null,

    @Column(nullable = false)
    var name: String? = null,

    @Column(unique = true)
    var tel: String? = null,

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    var memberRole: MemberRole? = null,

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    var memberStatus: MemberStatus? = null,
) : BaseEntity() {

    @OneToMany(mappedBy = "member", cascade = [CascadeType.ALL], orphanRemoval = true)
    val memberHospitals: MutableList<MemberHospital> = mutableListOf()

    fun addHospital(hospital: Hospital) {
        val memberHospital = MemberHospital(this, hospital)
        memberHospitals.add(memberHospital)
    }

    fun isMineHospital(hospitalId: Long): Boolean {
        return memberHospitals.any { it.hospital.id == hospitalId }
    }
}
