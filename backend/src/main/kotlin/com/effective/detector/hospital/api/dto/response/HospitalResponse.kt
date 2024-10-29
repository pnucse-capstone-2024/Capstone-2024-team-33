package com.effective.detector.hospital.api.dto.response

import com.effective.detector.hospital.domain.Hospital
import com.effective.detector.hospital.domain.HospitalType

data class HospitalResponse(
    val id: Long? = null,
    val name: String? = null,
    val tel: String? = null,
    val type: HospitalType? = null,
    val address: String? = null,
) {
    companion object {
        fun from(
            hospital: Hospital,
        ): HospitalResponse {
            return HospitalResponse(
                id = hospital.id!!,
                name = hospital.name,
                type = hospital.type,
                address = hospital.address,
                tel = hospital.tel,
            )
        }
    }
}
