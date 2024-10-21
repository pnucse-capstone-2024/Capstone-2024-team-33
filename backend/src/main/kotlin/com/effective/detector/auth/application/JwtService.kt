package com.effective.detector.auth.application

import com.effective.detector.member.domain.MemberRole
import io.jsonwebtoken.Claims
import io.jsonwebtoken.Jwts
import io.jsonwebtoken.SignatureAlgorithm
import io.jsonwebtoken.security.Keys
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.nio.charset.StandardCharsets
import java.util.*

@Service
class JwtService {

    @Value("\${jwt.secret}")
    private val secret: String? = null

    companion object {
        private const val ACCESS_EXPIRE_MILLIS = 1000 * 60 * 60 * 24 * 7 // 7 days
    }

    fun generateAccessToken(id: Long?, memberRole: MemberRole): String {
        return Jwts.builder()
            .setSubject(id.toString())
            .claim("role", memberRole.name)
            .setIssuedAt(Date())
            .setExpiration(Date(System.currentTimeMillis() + ACCESS_EXPIRE_MILLIS))
            .signWith(Keys.hmacShaKeyFor(secret?.toByteArray(StandardCharsets.UTF_8)), SignatureAlgorithm.HS256)
            .compact()
    }

    fun extractClaims(token: String?): Claims {
        val jwtParser = Jwts.parserBuilder()
            .setSigningKey(Keys.hmacShaKeyFor(secret?.toByteArray(StandardCharsets.UTF_8)))
            .build()
        return jwtParser.parseClaimsJws(token).body
    }
}
