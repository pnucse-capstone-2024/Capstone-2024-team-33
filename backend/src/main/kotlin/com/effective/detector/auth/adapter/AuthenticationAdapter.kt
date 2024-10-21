package com.effective.detector.auth.adapter

import com.effective.detector.member.domain.MemberRole

/**
 * JWT를 만들기 위해 필요한 id와 memberRole을 반환하는 인터페이스를 정의
 */
interface AuthenticationAdapter {

    fun getId(): Long?

    fun getMemberRole(): MemberRole
}
