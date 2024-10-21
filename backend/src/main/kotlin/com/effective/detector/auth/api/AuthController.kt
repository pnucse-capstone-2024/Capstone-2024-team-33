package com.effective.detector.auth.api

import com.effective.detector.auth.api.dto.LoginRequest
import com.effective.detector.auth.api.dto.LoginMemberResponse
import com.effective.detector.auth.api.dto.MemberResponse
import com.effective.detector.auth.api.dto.SignupRequest
import com.effective.detector.auth.application.AuthService
import com.effective.detector.common.annotation.LoginMember
import com.effective.detector.member.application.MemberService
import com.effective.detector.member.domain.Member
import io.swagger.v3.oas.annotations.Operation
import io.swagger.v3.oas.annotations.tags.Tag
import jakarta.servlet.http.HttpServletResponse
import jakarta.validation.Valid
import jakarta.validation.constraints.Pattern
import org.springframework.http.ResponseEntity
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.*

@Tag(name = "[Auth] 로그인 및 회원가입", description = "인증 관련 기능")
@RestController
@RequestMapping("/auth")
class AuthController(
    private val authService: AuthService,
    private val memberService: MemberService,
) {

    @Operation(summary = "회원가입 & 병원 등록")
    @PreAuthorize("permitAll()")
    @PostMapping("/signup")
    fun signup(@RequestBody @Valid request: SignupRequest): ResponseEntity<Void> {
        authService.signup(request)
        return ResponseEntity.ok().build()
    }

    @Operation(summary = "회원 탈퇴")
    @PreAuthorize("isAuthenticated()")
    @PostMapping("/signout")
    fun signOut(@LoginMember member: Member): ResponseEntity<Void> {
        authService.signOut(member)
        return ResponseEntity.noContent().build()
    }

    @Operation(summary = "로그인")
    @PreAuthorize("permitAll()")
    @PostMapping("/login")
    fun login(
        response: HttpServletResponse,
        @RequestBody @Valid loginDto: LoginRequest,
    ): ResponseEntity<LoginMemberResponse> {
        return ResponseEntity.ok(authService.login(response, loginDto))
    }

    @Operation(summary = "로그아웃")
    @PreAuthorize("permitAll()")
    @PostMapping("/logout")
    fun logout(response: HttpServletResponse): ResponseEntity<Void> {
        authService.logout(response)
        return ResponseEntity.ok().build()
    }

    @Operation(summary = "내 정보 조회")
    @PreAuthorize("isAuthenticated()")
    @GetMapping("/members/me")
    fun getMemberInfo(
        response: HttpServletResponse,
        @LoginMember member: Member,
    ): ResponseEntity<MemberResponse> {
        return ResponseEntity.ok(authService.getMemberInfo(response, member))
    }

    @Operation(summary = "아이디 중복 확인")
    @PreAuthorize("permitAll()")
    @GetMapping("/login-id/{login-id}")
    fun checkLoginId(
        @PathVariable("login-id") loginId: @Valid @Pattern(
            regexp = "^[a-zA-Z0-9_-]{8,32}$",
            message = "아이디는 8~32자의 영문 대소문자, 숫자, -, _만 사용 가능합니다."
        ) String?,
    ): ResponseEntity<Void> {
        memberService.checkLoginIdDuplicated(loginId)
        return ResponseEntity.ok().build()
    }
}
