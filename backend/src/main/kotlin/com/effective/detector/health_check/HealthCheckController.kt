package com.effective.detector.health_check

import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class HealthCheckController {

    @GetMapping("/health-check")
    fun healthCheck(): ResponseEntity<String> {
        return ResponseEntity.ok("ok")
    }

    @GetMapping("/cicd-test")
    fun cicdTest(): ResponseEntity<String> {
        return ResponseEntity.ok("cicd ok")
    }
}
