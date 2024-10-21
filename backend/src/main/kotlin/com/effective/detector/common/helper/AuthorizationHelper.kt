package com.effective.detector.common.helper

import com.effective.detector.common.error.BusinessError
import com.effective.detector.common.error.BusinessException
import com.effective.detector.member.domain.MemberRole
import org.springframework.security.core.GrantedAuthority
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.stereotype.Component


@Component
class AuthorizationHelper {

    fun isAdmin(): Boolean {
        val role: MemberRole = getMyRole()
        return role == MemberRole.ROLE_SUPER_ADMIN || role == MemberRole.ROLE_ADMIN
    }

    fun isSuperman(): Boolean {
        val role: MemberRole = getMyRole()
        return role == MemberRole.ROLE_SUPER_ADMIN
    }

    fun getMyId(): String {
        val authentication: org.springframework.security.core.Authentication =
            SecurityContextHolder.getContext().authentication
        return authentication.principal.toString()
    }

    fun getMyRole(): MemberRole {
        val authentication: org.springframework.security.core.Authentication =
            SecurityContextHolder.getContext().getAuthentication()
        val authorities: Collection<GrantedAuthority?> = authentication.authorities
        if (authorities.isEmpty()) {
            throw BusinessException(BusinessError.ROLE_NOT_FOUND)
        } else if (authorities.size > 1) {
            throw BusinessException(BusinessError.MULTIPLE_ROLE_FOUND)
        } else {
            val myRole = authorities.iterator().next()!!.authority
            try {
                return MemberRole.valueOf(myRole)
            } catch (e: IllegalArgumentException) {
                throw BusinessException(BusinessError.ROLE_TYPE_ERROR)
            }
        }
    }
}
