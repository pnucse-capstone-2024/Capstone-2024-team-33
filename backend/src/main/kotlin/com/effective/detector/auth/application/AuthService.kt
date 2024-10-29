package com.effective.detector.auth.application

import com.effective.detector.auth.adapter.AuthenticationAdapter
import com.effective.detector.auth.api.dto.LoginRequest
import com.effective.detector.auth.api.dto.LoginMemberResponse
import com.effective.detector.auth.api.dto.MemberResponse
import com.effective.detector.auth.api.dto.SignupRequest
import com.effective.detector.common.error.BusinessError
import com.effective.detector.common.error.BusinessException
import com.effective.detector.common.helper.AuthorizationHelper
import com.effective.detector.hospital.application.HospitalService
import com.effective.detector.member.application.MemberService
import com.effective.detector.member.domain.Member
import com.effective.detector.member.domain.MemberRepository
import com.effective.detector.member.domain.MemberRole
import com.effective.detector.member.domain.MemberStatus
import jakarta.servlet.http.HttpServletResponse
import org.springframework.security.authentication.AuthenticationManager
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.Authentication
import org.springframework.security.core.AuthenticationException
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

@Service
@Transactional(readOnly = true)
class AuthService(
    private val memberService: MemberService,
    private val passwordEncoder: PasswordEncoder,
    private val hospitalService: HospitalService,
    private val authenticationManager: AuthenticationManager,
    private val cookieService: CookieService,
    private val authorizationHelper: AuthorizationHelper,
    private val memberRepository: MemberRepository,
) {

    @Transactional
    fun signup(request: SignupRequest) {
        memberService.validatedLoginId(request.loginId!!)
        memberService.validatedTel(request.adminTel!!)
        val hospital = hospitalService.findHospital(request.hospitalId!!)
        hospital.initSlot()

        val member = Member(
            loginId = request.loginId,
            loginPassword = passwordEncoder.encode(request.loginPassword),
            name = request.adminName,
            tel = request.adminTel,
            memberRole = MemberRole.ROLE_ADMIN,
            memberStatus = MemberStatus.ACTIVE,
        )
        member.addHospital(hospital)
        memberService.save(member)
    }

    fun login(response: HttpServletResponse, loginDto: LoginRequest): LoginMemberResponse {
        val authentication: Authentication
        try {
            authentication = this.getAuthenticationFromIdPassword(
                loginDto.loginId, loginDto.loginPassword
            )
        } catch (exception: AuthenticationException) {
            throw BusinessException(BusinessError.ID_PASSWORD_AUTH_FAILED)
        }
        val adapter: AuthenticationAdapter = authentication.principal as AuthenticationAdapter
        val accessToken = cookieService.authenticate(adapter.getId(), adapter.getMemberRole(), response)
        return memberService.getLoginMemberById(adapter.getId(), accessToken)
    }

    private fun getAuthenticationFromIdPassword(loginId: String, loginPassword: String): Authentication {
        return authenticationManager.authenticate(UsernamePasswordAuthenticationToken(loginId, loginPassword))
    }

    fun getMemberInfo(response: HttpServletResponse, member: Member): MemberResponse? {
        if (authorizationHelper.getMyRole() != member.memberRole) {
            this.logout(response)
            throw BusinessException(BusinessError.MEMBER_ROLE_NOT_MATCHED)
        }
        return memberService.getMemberById(member.id)
    }

    fun logout(response: HttpServletResponse) {
        response.addCookie(cookieService.deleteAccessTokenCookie())
    }

    @Transactional
    fun signOut(member: Member) {
        memberRepository.delete(member)
    }
}
