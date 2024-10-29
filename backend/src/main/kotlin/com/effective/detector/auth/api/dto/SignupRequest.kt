package com.effective.detector.auth.api.dto

import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Pattern

data class SignupRequest(
    @field:Pattern(
        regexp = "^[a-zA-Z0-9_-]{8,32}$",
        message = "아이디는 8~32자의 영문 대소문자, 숫자, -, _만 사용 가능합니다."
    )
    val loginId: String?,

    @field:Pattern(
        regexp = "^(?=.*[A-Z])(?=.*[a-z])(?=.*\\d)[\\s\\S]{8,32}$",
        message = "비밀번호는 8~32자의 영문 대소문자, 숫자를 모두 포함하여 주세요."
    )
    val loginPassword: String?,

    @field:NotBlank(message = "관리자 이름을 입력해주세요.")
    val adminName: String?,

    @field:NotBlank(message = "관리자 전화번호를 입력해주세요.")
    @field:Pattern(
        regexp = "^010([0-9]{3,4})([0-9]{4})$",
        message = "휴대폰 번호 형식에 맞지 않습니다."
    )
    val adminTel: String?,

    @field:NotNull(message = "병원 Id를 입력해주세요.")
    val hospitalId: Long?,
)
