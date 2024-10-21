package com.effective.detector.auth.api.dto

import jakarta.validation.constraints.NotBlank

data class LoginRequest(
    @field:NotBlank
    val loginId: String,

    @field:NotBlank
    val loginPassword: String,
)
