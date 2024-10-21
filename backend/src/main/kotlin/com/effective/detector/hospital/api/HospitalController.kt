package com.effective.detector.hospital.api

import com.effective.detector.common.annotation.LoginMember
import com.effective.detector.hospital.api.dto.request.MonitorChangeRequest
import com.effective.detector.hospital.api.dto.response.AccidentMonthlyResponse
import com.effective.detector.hospital.api.dto.response.AccidentYearlyResponse
import com.effective.detector.hospital.api.dto.response.CameraResponse
import com.effective.detector.hospital.api.dto.response.HospitalResponse
import com.effective.detector.hospital.application.HospitalService
import com.effective.detector.hospital.application.ValidateService
import com.effective.detector.member.domain.Member
import io.swagger.v3.oas.annotations.Operation
import io.swagger.v3.oas.annotations.tags.Tag
import jakarta.validation.Valid
import org.springframework.http.ResponseEntity
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.web.bind.annotation.*

@Tag(name = "[Hospital] 병원", description = "병원 관련 기능")
@RestController
@RequestMapping("/hospitals")
class HospitalController(
    private val hospitalService: HospitalService,
    private val validateService: ValidateService,
) {

    @Operation(summary = "병원 검색")
    @PreAuthorize("permitAll()")
    @GetMapping
    fun findHospitalsByKeyword(
        @RequestParam(required = false) keyword: String?,
    ): ResponseEntity<List<HospitalResponse>> {
        return ResponseEntity.ok(hospitalService.findHospitalsByKeyword(keyword))
    }

    @Operation(summary = "병원 내 사고 건 수 통계 조회(월별)")
    @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
    @GetMapping("/{hospitalId}/statistics/month")
    fun getStatisticsByMonth(
        @PathVariable hospitalId: Long,
        @RequestParam(required = false) year: Int,
        @LoginMember member: Member,
    ): ResponseEntity<List<AccidentMonthlyResponse>> {
        validateService.checkMemberHospital(member, hospitalId)
        return ResponseEntity.ok(hospitalService.getStatisticsByMonth(hospitalId, year))
    }

    @Operation(summary = "병원 내 사고 건 수 통계 조회(연도별)")
    @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
    @GetMapping("/{hospitalId}/statistics/year")
    fun getStatisticsByYear(
        @PathVariable hospitalId: Long,
        @LoginMember member: Member,
    ): ResponseEntity<List<AccidentYearlyResponse>> {
        validateService.checkMemberHospital(member, hospitalId)
        return ResponseEntity.ok(hospitalService.getStatisticsByYear(hospitalId))
    }

    @Operation(summary = "병원에 설치된 카메라 목록 조회")
    @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
    @GetMapping("/{hospitalId}/cameras")
    fun getAllCameras(
        @PathVariable hospitalId: Long,
        @LoginMember member: Member,
    ): ResponseEntity<List<CameraResponse?>> {
        validateService.checkMemberHospital(member, hospitalId)
        return ResponseEntity.ok(hospitalService.getAllCameras(hospitalId))
    }

    @Operation(summary = "병원에서 모니터링하고 있는 카메라 목록 조회")
    @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
    @GetMapping("/{hospitalId}/monitors")
    fun getMonitoringCameras(
        @PathVariable hospitalId: Long,
        @LoginMember member: Member,
    ): ResponseEntity<List<CameraResponse?>> {
        validateService.checkMemberHospital(member, hospitalId)
        return ResponseEntity.ok(hospitalService.getMonitoringCameras(hospitalId))
    }

    @Operation(summary = "병원에서 모니터링하고 있는 카메라 변경")
    @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
    @PatchMapping("/{hospitalId}/monitors")
    fun updateMonitoringCamera(
        @PathVariable hospitalId: Long,
        @RequestBody @Valid request: MonitorChangeRequest,
        @LoginMember member: Member,
    ): ResponseEntity<Void> {
        validateService.checkMemberHospital(member, hospitalId)
        hospitalService.updateMonitoringCamera(hospitalId, request.slot, request.cameraId)
        return ResponseEntity.noContent().build()
    }
}
