package com.effective.detector.auth.adapter

import com.effective.detector.member.domain.Member
import com.effective.detector.member.domain.MemberRole
import com.effective.detector.member.domain.MemberStatus
import org.springframework.security.core.GrantedAuthority
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.userdetails.UserDetails

class LoginIdAdapterImpl(member: Member) : UserDetails, AuthenticationAdapter {
    private val id: Long? = member.id
    private val name: String? = member.name
    private val memberRole: MemberRole = member.memberRole ?: MemberRole.ROLE_ADMIN
    private val memberStatus: MemberStatus = member.memberStatus ?: MemberStatus.ACTIVE
    private val loginPassword: String? = member.loginPassword

    override fun getId(): Long? {
        return this.id
    }

    override fun getMemberRole(): MemberRole {
        return this.memberRole
    }

    override fun getAuthorities(): Collection<GrantedAuthority> {
        return listOf(SimpleGrantedAuthority(this.memberRole.name))
    }

    override fun getPassword(): String? {
        return this.loginPassword
    }

    override fun getUsername(): String? {
        return this.name
    }

    override fun isAccountNonExpired(): Boolean {
        return true
    }

    override fun isAccountNonLocked(): Boolean {
        return true
    }

    override fun isCredentialsNonExpired(): Boolean {
        return true
    }

    override fun isEnabled(): Boolean {
        return this.memberStatus == MemberStatus.ACTIVE
    }
}
