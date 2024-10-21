package com.effective.detector.common.config

import net.nurigo.java_sdk.api.Message
import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class CoolSmsConfig {

    @Value("\${coolsms.api-key}")
    private val apiKey: String? = null

    @Value("\${coolsms.secret-key}")
    private val secretKey: String? = null

    @Bean
    fun coolSmsSender(): Message {
        return Message(apiKey, secretKey)
    }
}
