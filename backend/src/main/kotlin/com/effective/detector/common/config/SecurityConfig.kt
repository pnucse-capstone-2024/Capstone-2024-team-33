package com.effective.detector.common.config

import com.effective.detector.member.domain.MemberRole.*

import com.effective.detector.common.filter.JwtAuthorizationFilter
import com.effective.detector.common.handler.JwtAccessDeniedHandler
import com.effective.detector.common.handler.JwtAuthenticationEntryPoint
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.http.HttpMethod
import org.springframework.security.authentication.AuthenticationManager
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity
import org.springframework.security.config.http.SessionCreationPolicy
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.security.web.SecurityFilterChain
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter
import org.springframework.web.cors.CorsConfigurationSource

@EnableMethodSecurity
@EnableWebSecurity
@Configuration
class SecurityConfig(
    private val jwtAuthorizationFilter: JwtAuthorizationFilter,
    private val jwtAccessDeniedHandler: JwtAccessDeniedHandler,
    private val jwtAuthenticationEntryPoint: JwtAuthenticationEntryPoint,
    private val corsConfigurationSource: CorsConfigurationSource,
) {

    @Bean
    fun passwordEncoder(): PasswordEncoder {
        return BCryptPasswordEncoder()
    }

    @Bean
    fun authenticationManager(config: AuthenticationConfiguration): AuthenticationManager {
        return config.authenticationManager
    }

    @Bean
    fun filterChain(http: HttpSecurity): SecurityFilterChain {
        http.csrf().disable()
            .cors().configurationSource(corsConfigurationSource).and()
            .formLogin().disable()
            .logout().disable()
            .sessionManagement()
            .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            .and()
            .authorizeHttpRequests()
            // AdminControllers: 메소드 어노테이션이 누락된 경우에도 어드민 경로 접근이 차단되도록 하는 방어 로직
            .requestMatchers(HttpMethod.POST, "/admin").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.GET, "/admin").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.PUT, "/admin").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.DELETE, "/admin").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.PATCH, "/admin").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.POST, "/admin/*").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.GET, "/admin/*").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.PUT, "/admin/*").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.DELETE, "/admin/*").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .requestMatchers(HttpMethod.PATCH, "/admin/*").hasAnyAuthority(ROLE_ADMIN.name, ROLE_SUPER_ADMIN.name)
            .anyRequest().permitAll()
            .and()
            .exceptionHandling()
            .accessDeniedHandler(jwtAccessDeniedHandler)
            .authenticationEntryPoint(jwtAuthenticationEntryPoint)
            .and()
            .addFilterBefore(jwtAuthorizationFilter, UsernamePasswordAuthenticationFilter::class.java)

        return http.build()
    }
}
