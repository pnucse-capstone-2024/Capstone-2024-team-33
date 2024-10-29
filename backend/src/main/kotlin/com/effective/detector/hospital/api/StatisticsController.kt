package com.effective.detector.hospital.api

import com.effective.detector.common.annotation.LoginMember
import com.effective.detector.hospital.api.dto.response.AccidentMonthlyResponse
import com.effective.detector.hospital.api.dto.response.AccidentYearlyResponse
import com.effective.detector.hospital.api.dto.response.AllPerformanceStatisticResponse
import com.effective.detector.hospital.api.dto.response.HospitalPerformanceStatisticResponse
import com.effective.detector.hospital.application.AccidentService
import com.effective.detector.hospital.application.ValidateService
import com.effective.detector.member.domain.Member
import io.swagger.v3.oas.annotations.Operation
import io.swagger.v3.oas.annotations.tags.Tag
import org.springframework.http.ResponseEntity
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@Tag(name = "[Statistics] 병원 통계", description = "병원 통계 관련 기능")
@RestController
@RequestMapping("/statistics")
class StatisticsController(
    private val accidentService: AccidentService,
    private val validateService: ValidateService,
) {

    @Operation(summary = "전국사고 건 수 통계 조회(월별)")
    @PreAuthorize("permitAll()")
    @GetMapping("/month")
    fun getStatisticsByMonth(
        @RequestParam(required = false) year: Int,
    ): ResponseEntity<List<AccidentMonthlyResponse>> {
        return ResponseEntity.ok(accidentService.getStatisticsByMonth(year))
    }

    @Operation(summary = "전국사고 건 수 통계 조회(연도별)")
    @PreAuthorize("permitAll()")
    @GetMapping("/year")
    fun getStatisticsByYear(): ResponseEntity<List<AccidentYearlyResponse>> {
        return ResponseEntity.ok(accidentService.getStatisticsByYear())
    }

    @Operation(summary = "통계가 존재하는 연도 목록 조회")
    @PreAuthorize("permitAll()")
    @GetMapping("/exist/years")
    fun getYearByExistAccident(): ResponseEntity<List<Int>> {
        return ResponseEntity.ok(accidentService.getYearByExistAccident())
    }

    @Operation(summary = "병원 내 통계가 존재하는 연도 목록 조회")
    @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
    @GetMapping("/exist/years/hospital/{hospitalId}")
    fun getYearByExistAccidentHospital(
        @PathVariable hospitalId: Long,
        @LoginMember member: Member,
    ): ResponseEntity<List<Int>> {
        validateService.checkMemberHospital(member, hospitalId)
        return ResponseEntity.ok(accidentService.getYearByExistAccidentHospital(hospitalId))
    }

    @Operation(summary = "전국 사고 성과 통계 조회")
    @PreAuthorize("permitAll()")
    @GetMapping("/performance")
    fun getPerformance(
    ): ResponseEntity<AllPerformanceStatisticResponse> {
        return ResponseEntity.ok(accidentService.getPerformance())
    }

    @Operation(summary = "병원 내 사고 성과 통계 조회")
    @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
    @GetMapping("/performance/hospitals/{hospitalId}")
    fun getPerformanceByHospital(
        @PathVariable hospitalId: Long,
        @LoginMember member: Member,
    ): ResponseEntity<HospitalPerformanceStatisticResponse> {
        validateService.checkMemberHospital(member, hospitalId)
        return ResponseEntity.ok(accidentService.getPerformanceByHospital(hospitalId))
    }
}
