package com.effective.detector

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration
import org.springframework.boot.runApplication
import org.springframework.data.jpa.repository.config.EnableJpaAuditing

@EnableJpaAuditing
@SpringBootApplication(exclude = [SecurityAutoConfiguration::class])
class DetectorApplication

fun main(args: Array<String>) {
    runApplication<DetectorApplication>(*args)
}
