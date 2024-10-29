package com.effective.detector.common.config

import org.springframework.context.annotation.Configuration
import org.springframework.messaging.simp.config.MessageBrokerRegistry
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker
import org.springframework.web.socket.config.annotation.StompEndpointRegistry
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer

@Configuration
@EnableWebSocketMessageBroker
class WebSocketStompConfig : WebSocketMessageBrokerConfigurer {

    override fun configureMessageBroker(config: MessageBrokerRegistry) {
        config.enableSimpleBroker("/ws/topic") // 클라이언트가 구독할 수 있는 목적지(prefix)를 설정합니다.
        config.setApplicationDestinationPrefixes("/ws/app") // 메시지를 보낼 때 사용하는 prefix를 설정합니다.
    }

    override fun registerStompEndpoints(registry: StompEndpointRegistry) {
        registry.addEndpoint("/ws")
            .setAllowedOrigins(
                "http://localhost:3000",
                "https://teameffective.link"
            )
            .withSockJS()
    }
}
