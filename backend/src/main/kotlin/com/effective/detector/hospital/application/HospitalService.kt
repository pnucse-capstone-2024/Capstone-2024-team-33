package com.effective.detector.hospital.application

import com.effective.detector.common.error.BusinessError
import com.effective.detector.common.error.BusinessException
import com.effective.detector.common.util.findByIdOrThrow
import com.effective.detector.hospital.api.dto.response.AccidentMonthlyResponse
import com.effective.detector.hospital.api.dto.response.AccidentYearlyResponse
import com.effective.detector.hospital.api.dto.response.HospitalResponse
import com.effective.detector.hospital.api.dto.response.CameraResponse
import com.effective.detector.hospital.domain.*
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

@Service
@Transactional(readOnly = true)
class HospitalService(
    private val hospitalRepository: HospitalRepository,
    private val accidentRepository: AccidentRepository,
    private val cameraRepository: CameraRepository,
) {

    fun findHospital(id: Long): Hospital {
        return hospitalRepository.findByIdOrThrow(id)
    }

    fun findHospitalsByKeyword(keyword: String?): List<HospitalResponse> {
        return if (keyword.isNullOrBlank())
            hospitalRepository.findAll().map { HospitalResponse.from(it) }
        else
            hospitalRepository.findByNameContaining(keyword).map { HospitalResponse.from(it) }
    }

    fun getStatisticsByMonth(hospitalId: Long, year: Int): List<AccidentMonthlyResponse> {
        val accidents: List<Accident> = accidentRepository.findAllByYearAndHospitalId(year, hospitalId)
        return accidents.groupBy { it.startTime.month }
            .map { (month, accidentsInMonth) ->
                val typeCounts = accidentsInMonth.groupingBy { it.type ?: AccidentType.ETC }.eachCount()
                AccidentMonthlyResponse(
                    month = "${month.value}월",
                    slipping = typeCounts[AccidentType.SLIPPIING] ?: 0,
                    fighting = typeCounts[AccidentType.FIGHTING] ?: 0,
                    poorFacilities = typeCounts[AccidentType.POOR_FACILITIES] ?: 0,
                    carelessness = typeCounts[AccidentType.CARELESSNESS] ?: 0,
                    malfunction = typeCounts[AccidentType.MALFUNCTION] ?: 0,
                    etc = typeCounts[AccidentType.ETC] ?: 0,
                    total = accidentsInMonth.size
                )
            }
    }

    fun getStatisticsByYear(hospitalId: Long): List<AccidentYearlyResponse> {
        val accidents = accidentRepository.findAllByHospitalId(hospitalId)
        return accidents.groupBy { it.startTime.year }
            .map { (year, accidentsInYear) ->
                val typeCounts = accidentsInYear.groupingBy { it.type ?: AccidentType.ETC }.eachCount()
                AccidentYearlyResponse(
                    year = year,
                    slipping = typeCounts[AccidentType.SLIPPIING] ?: 0,
                    fighting = typeCounts[AccidentType.FIGHTING] ?: 0,
                    poorFacilities = typeCounts[AccidentType.POOR_FACILITIES] ?: 0,
                    carelessness = typeCounts[AccidentType.CARELESSNESS] ?: 0,
                    malfunction = typeCounts[AccidentType.MALFUNCTION] ?: 0,
                    etc = typeCounts[AccidentType.ETC] ?: 0,
                    total = accidentsInYear.size
                )
            }
    }

    @Transactional
    fun updateMonitoringCamera(hospitalId: Long, slotId: Int, cameraId: Long) {
        val hospital = hospitalRepository.findByIdOrThrow(hospitalId)
        val slot = hospital.findSlot(slotId)

        if (cameraId == -1L) {
            slot.remove()
            return
        }
        val camera = cameraRepository.findByIdOrThrow(cameraId)
        if (hospital.hasSlot(camera)) {
            throw BusinessException(BusinessError.SLOT_CAMERA_DUPLICATED)
        }
        slot.change(camera)
    }

    fun getMonitoringCameras(hospitalId: Long): List<CameraResponse?> {
        val hospital = hospitalRepository.findByIdOrThrow(hospitalId)
        return hospital.slots.map { CameraResponse.from(it.camera) }
    }

    fun getAllCameras(hospitalId: Long): List<CameraResponse> {
        val hospital = hospitalRepository.findByIdOrThrow(hospitalId)
        return hospital.cameras.map { CameraResponse.toDto(it) }
    }
}
