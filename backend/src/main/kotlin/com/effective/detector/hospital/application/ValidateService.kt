package com.effective.detector.hospital.application

import com.effective.detector.common.error.BusinessError
import com.effective.detector.common.error.BusinessException
import com.effective.detector.common.util.findByIdOrThrow
import com.effective.detector.hospital.domain.Accident
import com.effective.detector.hospital.domain.AccidentRepository
import com.effective.detector.member.domain.Member
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

@Service
@Transactional(readOnly = true)
class ValidateService(
    private val accidentRepository: AccidentRepository,
) {

    fun checkMemberHospital(member: Member, hospitalId: Long) {
        if (!member.isMineHospital(hospitalId)) {
            throw BusinessException(BusinessError.IS_NOT_MY_HOSPITAL)
        }
    }

    fun checkMemberHospitalAccident(member: Member, accidentId: Long) {
        val accident = accidentRepository.findByIdOrThrow(accidentId)
        member.isMineHospital(accident.hospital.id!!)
    }
}
