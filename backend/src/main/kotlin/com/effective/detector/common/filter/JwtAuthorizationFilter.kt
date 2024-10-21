package com.effective.detector.common.filter

import com.effective.detector.auth.application.JwtService
import com.effective.detector.common.util.logWarn
import io.jsonwebtoken.Claims
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.lang.NonNull
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter

@Component
class JwtAuthorizationFilter(
    private val jwtService: JwtService,
) : OncePerRequestFilter() {

    override fun doFilterInternal(
        request: HttpServletRequest, @NonNull response: HttpServletResponse,
        @NonNull filterChain: FilterChain,
    ) {
        request.getHeaders("Authorization").toList().forEach {
            if (it.startsWith("Bearer ")) {
                val token = it.substring(7)
                try {
                    val claims: Claims = jwtService.extractClaims(token)
                    val id = claims.subject
                    val role = claims["role"] as String?
                    val authenticationToken = UsernamePasswordAuthenticationToken(
                        id, null, listOf(SimpleGrantedAuthority(role))
                    )
                    SecurityContextHolder.getContext().authentication = authenticationToken
                } catch (e: RuntimeException) {
                    logWarn(e.message, e)
                }
            }
        }

        // 필터 체인을 통해 요청과 응답을 전달
        filterChain.doFilter(request, response)
    }
}
