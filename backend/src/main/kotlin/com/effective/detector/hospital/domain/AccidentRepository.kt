package com.effective.detector.hospital.domain

import com.effective.detector.hospital.api.dto.response.PrimaryReasonResponse
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param

interface AccidentRepository : JpaRepository<Accident, Long> {
    @Query("SELECT a FROM Accident a WHERE YEAR(a.startTime) = :year")
    fun findAllByYear(@Param("year") year: Int): List<Accident>

    @Query(
        "SELECT a FROM Accident a " +
                "WHERE YEAR(a.startTime) = :year " +
                "AND a.hospital.id = :hospitalId"
    )
    fun findAllByYearAndHospitalId(
        @Param("year") year: Int,
        @Param("hospitalId") hospitalId: Long,
    ): List<Accident>

    @Query("SELECT a FROM Accident a WHERE a.hospital.id = :hospitalId")
    fun findAllByHospitalId(
        @Param("hospitalId") hospitalId: Long,
    ): List<Accident>

    @Query("SELECT a FROM Accident a WHERE a.hospital.id = :hospitalId")
    fun findAllByHospitalId(
        @Param("hospitalId") hospitalId: Long,
        pageable: Pageable,
    ): Page<Accident>

    @Query(
        "SELECT a FROM Accident a " +
                "WHERE a.hospital.id = :hospitalId " +
                "AND a.isProcess = false"
    )
    fun findAllByHospitalIdAndUnprocess(@Param("hospitalId") hospitalId: Long): List<Accident>

    @Query("SELECT DISTINCT EXTRACT(YEAR FROM a.startTime) FROM Accident a ORDER BY EXTRACT(YEAR FROM a.startTime) DESC")
    fun findDistinctYears(): List<Int>

    @Query("SELECT DISTINCT EXTRACT(YEAR FROM a.startTime) FROM Accident a WHERE a.hospital.id = :hospitalId ORDER BY EXTRACT(YEAR FROM a.startTime) DESC")
    fun findDistinctYearsByHospitalId(hospitalId: Long): List<Int>

    @Query(
        """
        SELECT COUNT(a) 
        FROM Accident a
        WHERE YEAR(a.startTime) = :year
    """
    )
    fun getTotalAccidentCountForYear(year: Int): Long

    @Query("SELECT COUNT(a) FROM Accident a WHERE a.hospital.id = :hospitalId AND YEAR(a.startTime) = :year")
    fun getTotalAccidentCountByHospitalId(hospitalId: Long, year: Int): Long

    @Query(
        """
        SELECT new com.effective.detector.hospital.api.dto.response.PrimaryReasonResponse(a.type)
        FROM Accident a
        WHERE YEAR(a.startTime) = :year
        GROUP BY a.type
        ORDER BY COUNT(a) DESC
    """
    )
    fun getPrimaryReasonForAccidents(year: Int): List<PrimaryReasonResponse>

    @Query(
        """
        SELECT new com.effective.detector.hospital.api.dto.response.PrimaryReasonResponse(a.type)
        FROM Accident a
        WHERE a.hospital.id = :hospitalId AND YEAR(a.startTime) = :year
        GROUP BY a.type
        ORDER BY COUNT(a) DESC
        LIMIT 1
    """
    )
    fun getPrimaryReasonForAccidentsByHospitalId(hospitalId: Long, year: Int): List<PrimaryReasonResponse>

    @Query(
        """
        SELECT 
            YEAR(a.startTime), MONTH(a.startTime)
        FROM Accident a
        WHERE YEAR(a.startTime) = :year
        GROUP BY YEAR(a.startTime), MONTH(a.startTime)
        ORDER BY COUNT(a) DESC
    """
    )
    fun getMostAccidentsOccurredByMonthAndYear(year: Int): List<Array<Int>>

    @Query(
        """
        SELECT 
            YEAR(a.startTime), MONTH(a.startTime)
        FROM Accident a
        WHERE a.hospital.id = :hospitalId AND YEAR(a.startTime) = :year
        GROUP BY YEAR(a.startTime), MONTH(a.startTime)
        ORDER BY COUNT(a) DESC
    """
    )
    fun getMostAccidentsOccurredByMonthAndYearByHospitalId(hospitalId: Long, year: Int): List<Array<Any>>

    @Query("SELECT COUNT(a) FROM Accident a WHERE a.type = 'MALFUNCTION' AND YEAR(a.startTime) = :year")
    fun getMalfunctionAccidentCount(year: Int): Long

    @Query("SELECT COUNT(a) FROM Accident a WHERE a.type = 'MALFUNCTION' AND a.hospital.id = :hospitalId AND YEAR(a.startTime) = :year")
    fun getMalfunctionAccidentCountByHospitalId(hospitalId: Long, year: Int): Long
}
