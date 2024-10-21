package com.effective.detector.auth.application

import com.effective.detector.auth.adapter.LoginIdAdapterImpl
import com.effective.detector.common.error.BusinessError
import com.effective.detector.member.domain.MemberRepository
import org.springframework.security.core.userdetails.UserDetails
import org.springframework.security.core.userdetails.UserDetailsService
import org.springframework.security.core.userdetails.UsernameNotFoundException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

@Service
class MemberDetailsService(
    private val memberRepository: MemberRepository,
) : UserDetailsService {

    @Transactional(readOnly = true)
    override fun loadUserByUsername(loginId: String): UserDetails {
        val member = memberRepository.findByLoginId(loginId)
            ?: throw UsernameNotFoundException(BusinessError.MEMBER_NOT_FOUND.message)
        return LoginIdAdapterImpl(member)
    }
}
