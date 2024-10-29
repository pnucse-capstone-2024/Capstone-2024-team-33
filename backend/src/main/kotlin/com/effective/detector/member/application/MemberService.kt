package com.effective.detector.member.application

import com.effective.detector.auth.api.dto.LoginMemberResponse
import com.effective.detector.auth.api.dto.MemberResponse
import com.effective.detector.common.error.BusinessError
import com.effective.detector.common.error.BusinessException
import com.effective.detector.common.util.findByIdOrThrow
import com.effective.detector.hospital.api.dto.response.HospitalResponse
import com.effective.detector.hospital.domain.Hospital
import com.effective.detector.member.domain.Member
import com.effective.detector.member.domain.MemberRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

@Service
@Transactional(readOnly = true)
class MemberService(
    private val memberRepository: MemberRepository,
) {
    fun validatedLoginId(loginId: String) {
        memberRepository.findByLoginId(loginId)?.let {
            throw BusinessException(BusinessError.MEMBER_LOGIN_ID_DUPLICATED)
        }
    }

    fun validatedTel(adminTel: String) {
        memberRepository.findByTel(adminTel)?.let {
            throw BusinessException(BusinessError.MEMBER_TEL_DUPLICATED)
        }
    }

    @Transactional
    fun save(member: Member) {
        memberRepository.save(member)
    }

    fun getLoginMemberById(id: Long?, accessToken: String?): LoginMemberResponse {
        val member = memberRepository.findById(id!!).orElseThrow {
            BusinessException(BusinessError.ID_NOT_FOUND)
        }
        return LoginMemberResponse(
            id = member.id,
            name = member.memberHospitals.first().hospital.name,
            memberRole = member.memberRole,
            accessToken = accessToken,
            hospitalId = member.memberHospitals.map { it.hospital.id!! },
        )
    }

    fun getById(id: Long): Member {
        return memberRepository.findByIdOrThrow(id)
    }

    fun checkLoginIdDuplicated(loginId: String?) {
        if (memberRepository.existsByLoginId(loginId)) {
            throw BusinessException(BusinessError.LOGIN_ID_DUPLICATED)
        }
    }

    fun getMemberById(id: Long?): MemberResponse? {
        val member = memberRepository.findById(id!!).orElseThrow {
            BusinessException(BusinessError.ID_NOT_FOUND)
        }
        return MemberResponse(
            id = member.id,
            loginId = member.loginId,
            name = member.name,
            memberRole = member.memberRole,
            tel = member.tel,
            hospital = mapToDto(member.memberHospitals.first().hospital),
        )
    }

    fun mapToDto(hospital: Hospital): HospitalResponse {
        return HospitalResponse(
            id = hospital.id,
            name = hospital.name,
            tel = hospital.tel,
            type = hospital.type,
            address = hospital.address,
        )
    }
}
