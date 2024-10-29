package com.effective.detector.auth.application

import com.effective.detector.member.domain.MemberRole
import jakarta.servlet.http.Cookie
import jakarta.servlet.http.HttpServletResponse
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service

@Service
class CookieService(
    private val jwtService: JwtService,
) {
    @Value("\${cookie.domain}")
    private lateinit var domain: String

    fun authenticate(id: Long?, memberRole: MemberRole, response: HttpServletResponse): String {
        val accessToken: String = jwtService.generateAccessToken(id, memberRole)
        response.addCookie(this.makeAccessTokenCookie(accessToken))
        return accessToken
    }

    private fun makeAccessTokenCookie(token: String): Cookie {
        return this.makeCookie("access_token", token, 604800)
    }

    private fun makeCookie(key: String, value: String?, maxAge: Int): Cookie {
        val cookie = Cookie(key, value)
        cookie.secure = true
        cookie.isHttpOnly = true
        cookie.path = "/"
        cookie.domain = domain
        cookie.maxAge = maxAge
        return cookie
    }

    fun deleteAccessTokenCookie(): Cookie? {
        return this.makeCookie("access_token", null, 0)
    }
}
