package com.effective.detector.member.domain

import org.springframework.data.jpa.repository.JpaRepository

interface MemberRepository : JpaRepository<Member, Long> {

    fun findByLoginId(loginId: String): Member?

    fun findByTel(tel: String): Member?

    fun existsByLoginId(loginId: String?): Boolean
}
