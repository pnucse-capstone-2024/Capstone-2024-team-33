package com.effective.detector.hospital.application

import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.PutObjectRequest
import com.effective.detector.common.util.findByIdOrThrow
import com.effective.detector.common.util.logError
import com.effective.detector.common.util.logInfo
import com.effective.detector.hospital.api.dto.response.*
import com.effective.detector.hospital.domain.*
import com.effective.detector.image.application.*
import net.nurigo.java_sdk.exceptions.CoolsmsException
import org.jcodec.api.SequenceEncoder
import org.jcodec.common.io.NIOUtils
import org.jcodec.common.model.ColorSpace
import org.jcodec.common.model.Picture
import org.jcodec.common.model.Rational
import org.jcodec.scale.AWTUtil
import org.springframework.amqp.rabbit.core.RabbitTemplate
import org.springframework.beans.factory.annotation.Value
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.messaging.simp.SimpMessagingTemplate
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.File
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.UUID.randomUUID
import javax.imageio.ImageIO

@Service
@Transactional(readOnly = true)
class AccidentService(
    private val accidentRepository: AccidentRepository,
    private val amazonS3Client: AmazonS3,
    private val hospitalRepository: HospitalRepository,
    private val cameraRepository: CameraRepository,
    private val messagingTemplate: SimpMessagingTemplate,
    private val coolSmsSender: net.nurigo.java_sdk.api.Message,
    private val rabbitTemplate: RabbitTemplate,
    private val mikeRepository: MikeRepository,
) {

    @Value("\${video-prefix}")
    private val videoPrefix: String? = null

    @Value("\${coolsms.sender-number}")
    private val senderNumber: String? = null

    companion object {
        const val S3_BUCKET = "effective-bucket"
    }

    fun getStatisticsByMonth(year: Int): List<AccidentMonthlyResponse> {
        val accidents: List<Accident> = accidentRepository.findAllByYear(year)
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
                    total = accidentsInMonth.size,
                )
            }
    }

    fun getStatisticsByYear(): List<AccidentYearlyResponse> {
        val accidents = accidentRepository.findAll()
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

    fun getAll(pageable: Pageable): Page<AccidentResponse> {
        return accidentRepository.findAll(pageable).map { AccidentResponse.from(it) }
    }

    fun getAllByHospital(hospitalId: Long, pageable: Pageable): Page<AccidentResponse>? {
        return accidentRepository.findAllByHospitalId(hospitalId, pageable)
            .map { AccidentResponse.from(it) }
    }

    fun getYearByExistAccident(): List<Int> {
        return accidentRepository.findDistinctYears()
    }

    @Transactional
    fun update(accidentId: Long, typeId: Long, ageId: Long) {
        val accident = accidentRepository.findByIdOrThrow(accidentId)
        accident.update(AccidentType.from(typeId), AgeType.from(ageId))
        accident.process()
    }

    fun getUnprocessAccident(hospitalId: Long): List<UnprocessAccidentResponse> {
        return accidentRepository.findAllByHospitalIdAndUnprocess(hospitalId).map { UnprocessAccidentResponse.from(it) }
    }

    fun getYearByExistAccidentHospital(hospitalId: Long): List<Int> {
        return accidentRepository.findDistinctYearsByHospitalId(hospitalId)
    }

    fun getPerformance(): AllPerformanceStatisticResponse {
        val currentYear = LocalDate.now().year
        val lastYear = currentYear.dec()
        val totalAccidentCount = accidentRepository.getTotalAccidentCountForYear(currentYear)
        val lastYearAccidentCount = accidentRepository.getTotalAccidentCountForYear(lastYear)

        val primaryReason = accidentRepository.getPrimaryReasonForAccidents(currentYear)[0]

        val increaseRateByLastYear = if (lastYearAccidentCount != 0L) {
            ((totalAccidentCount - lastYearAccidentCount).toDouble() / lastYearAccidentCount) * 100
        } else {
            100.0 // 전년도 사고가 없을 경우 100% 증가로 간주
        }

        val malfunctionAccidentCount = accidentRepository.getMalfunctionAccidentCount(currentYear)
        val detectionAccuracy = if (totalAccidentCount > 0) {
            if (malfunctionAccidentCount == 0L) {
                100.0
            } else {
                (malfunctionAccidentCount.toDouble() / totalAccidentCount) * 100
            }
        } else {
            0.0
        }

        val mostAccidentsData = accidentRepository.getMostAccidentsOccurredByMonthAndYear(currentYear)
        val mostAccidentsOrccuredMonth = (mostAccidentsData.firstOrNull()?.get(1) ?: 1)
        val mostAccidentsOrccuredYear = (mostAccidentsData.firstOrNull()?.get(0) ?: currentYear)

        return AllPerformanceStatisticResponse(
            totalAccidentCount = totalAccidentCount,
            primaryReason = primaryReason,
            increaseRateByLastYear = increaseRateByLastYear,
            detectionAccuracy = detectionAccuracy,
            mostAccidentsOrccuredMonth = mostAccidentsOrccuredMonth,
            mostAccidentsOrccuredYear = mostAccidentsOrccuredYear
        )
    }

    fun getPerformanceByHospital(hospitalId: Long): HospitalPerformanceStatisticResponse {
        val currentYear = LocalDate.now().year
        val lastYear = currentYear.dec()
        val totalAccidentCount = accidentRepository.getTotalAccidentCountByHospitalId(hospitalId, currentYear)
        val lastYearAccidentCount = accidentRepository.getTotalAccidentCountByHospitalId(hospitalId, lastYear)

        val primaryReason = accidentRepository.getPrimaryReasonForAccidentsByHospitalId(hospitalId, currentYear)[0]

        val increaseRateByLastYear = if (lastYearAccidentCount != 0L) {
            ((totalAccidentCount - lastYearAccidentCount).toDouble() / lastYearAccidentCount) * 100
        } else {
            100.0 // 전년도 사고가 없을 경우 100% 증가로 간주
        }

        val malfunctionAccidentCount =
            accidentRepository.getMalfunctionAccidentCountByHospitalId(hospitalId, currentYear)
        val detectionAccuracy = if (totalAccidentCount > 0) {
            if (malfunctionAccidentCount == 0L) {
                100.0
            } else {
                (malfunctionAccidentCount.toDouble() / totalAccidentCount) * 100
            }
        } else {
            0.0
        }

        val mostAccidentsData =
            accidentRepository.getMostAccidentsOccurredByMonthAndYearByHospitalId(hospitalId, currentYear)
        val mostAccidentsOrccuredMonth = (mostAccidentsData.firstOrNull()?.get(1) ?: 1) as Int

        val allAccidentCount = accidentRepository.getTotalAccidentCountForYear(currentYear)
        val increaseRateByAverage = if (allAccidentCount > 0) {
            (totalAccidentCount.toDouble() / allAccidentCount) * 100
        } else {
            0.0
        }

        return HospitalPerformanceStatisticResponse(
            totalAccidentCount = totalAccidentCount,
            primaryReason = primaryReason,
            increaseRateByLastYear = increaseRateByLastYear,
            detectionAccuracy = detectionAccuracy,
            mostAccidentsOrccuredMonth = mostAccidentsOrccuredMonth,
            increaseRateByAverage = increaseRateByAverage,
        )
    }

    fun getAges(): List<AgeResponse> {
        return AgeType.entries.map { AgeResponse.from(it) }
    }

    fun getTypes(): List<TypeResponse> {
        return AccidentType.entries.map { TypeResponse.from(it) }
    }

    @Transactional
    fun thermalAccident(queueName: String, hospitalId: Long, cameraId: Long) {
        try {
            val outputFilePath = "${System.getProperty("user.dir")}/${randomUUID()}-video.mp4"
            val timeInfo = convertImageToVideo(queueName, outputFilePath) ?: return // 30초 내의 이미지들을 영상으로 변환
            val camera = cameraRepository.findByIdOrThrow(cameraId)
            val videoUrl = uploadToStorage(hospitalId, camera, timeInfo, outputFilePath) // 영상을 S3에 저장하고 DB 테이블에 정보들 저장.
            //sendVideoMessageToUser(hospitalId, camera, videoUrl) // 사고 발생 영상을 문자로 전송. TODO: 시연시 활성화.
            sendClientMessage("/ws/topic/accident/hospitals/$hospitalId", hospitalId, camera) // 클라이언트로 사고 발생 알림 보내기
        } catch (exception: Exception) {
            logError("RabbitMqMessage Handling Error", exception)
        }
    }

    private fun convertImageToVideo(queueName: String, outputFilePath: String): Pair<LocalDateTime?, LocalDateTime?>? {
        val imageList = mutableListOf<ByteArray>()
        var startTime: LocalDateTime? = null
        var endTime: LocalDateTime? = null
        val formatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME // 기본 ISO 형식 포맷터
        var message = rabbitTemplate.receive(queueName)
        while (message != null) {
            val imageMessageDto = rabbitTemplate.messageConverter.fromMessage(message) as ImageMessageDto
            if (startTime == null) {
                startTime = LocalDateTime.parse(imageMessageDto.time, formatter)
            }
            endTime = LocalDateTime.parse(imageMessageDto.time, formatter)
            imageList.add(imageMessageDto.bytes)
            message = rabbitTemplate.receive(queueName)
        }

        if (imageList.isNotEmpty()) {
            logInfo("Converting images to video...")
            createMp4(imageList, outputFilePath)
            return Pair(startTime, endTime)
        } else {
            logInfo("No images to process.")
        }
        return null
    }

    private fun createMp4(imageList: List<ByteArray>, outputFilePath: String) {
        val out = NIOUtils.writableFileChannel(outputFilePath)
        val encoder = SequenceEncoder.createWithFps(out, Rational.R(10, 1))  // FPS 설정

        for (imageBytes in imageList) {
            val img = ImageIO.read(ByteArrayInputStream(imageBytes))
            val resizedImage = resizeImage(img, img.width * 2, img.height * 2)
            encoder.encodeNativeFrame(bufferedImageToPicture(resizedImage))
        }

        encoder.finish()
        out.close()
    }

    private fun resizeImage(originalImage: BufferedImage, targetWidth: Int, targetHeight: Int): BufferedImage {
        val resizedImage = BufferedImage(targetWidth, targetHeight, originalImage.type)
        val g2d: Graphics2D = resizedImage.createGraphics()

        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

        g2d.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null)
        g2d.dispose()

        return resizedImage
    }

    private fun bufferedImageToPicture(image: BufferedImage): Picture {
        return AWTUtil.fromBufferedImage(image, ColorSpace.RGB)
    }

    private fun uploadToStorage(
        hospitalId: Long,
        camera: Camera,
        timeInfo: Pair<LocalDateTime?, LocalDateTime?>,
        outputFilePath: String,
    ): String {
        val videoUrl = "$hospitalId-${camera.id}-${LocalDateTime.now()}.mp4"
        uploadVideo(videoUrl, outputFilePath)
        saveVideoInfo(videoUrl, hospitalId, camera, timeInfo.first!!, timeInfo.second!!)
        return videoPrefix + videoUrl
    }

    private fun uploadVideo(videoUrl: String, outputFilePath: String) {
        val file = File(outputFilePath)
        amazonS3Client.putObject(
            PutObjectRequest(
                S3_BUCKET,
                videoUrl,
                file
            )
        )
        file.delete()
        logInfo("File Uploaded Successfully To S3!")
    }

    private fun saveVideoInfo(
        videoUrl: String,
        hospitalId: Long,
        camera: Camera,
        startTime: LocalDateTime,
        endTime: LocalDateTime,
    ) {
        val hospital = hospitalRepository.findByIdOrThrow(hospitalId) // 존재하는 병원인지 유효성 체크.
        accidentRepository.save(
            Accident(
                videoUrl = videoUrl,
                startTime = startTime,
                endTime = endTime,
                camera = camera,
                hospital = hospital
            )
        )
    }

    private fun sendVideoMessageToUser(hospitalId: Long, camera: Camera, videoUrl: String) {
        val hospital = hospitalRepository.findByIdOrThrow(hospitalId)

        val params = HashMap<String, String>()
        hospital.memberHospitals.map { it.member }.forEach {
            params["to"] = senderNumber!!
            params["from"] = it.tel!!
            params["type"] = "LMS"
            params["text"] = "${hospital.name}병원의 ${camera.name}번 카메라에서 낙상사고가 감지되었습니다. 영상을 확인해주세요. 영상 링크 : $videoUrl"
            params["app_version"] = "test app 1.2"

            try {
                coolSmsSender.send(params)
                logInfo("SMS Sent Successfully To ${it.tel}")
            } catch (e: CoolsmsException) {
                logError(e.message, e)
            }
        }
    }

    private fun sendClientMessage(topicPath: String, hospitalId: Long, camera: Camera) {
        logInfo("Send WebSocket STOMP Thermal Message To Path: $topicPath")
        messagingTemplate.convertAndSend(
            topicPath,
            AccidentMessageDto(
                hospitalId = hospitalId,
                camera = CameraMessageDto(
                    id = camera.id!!,
                    content = camera.name
                ),
                startTime = LocalDateTime.now(),
                endTime = LocalDateTime.now()
            )
        )
    }

    @Transactional
    fun audioAccident(hospitalId: Long, mikeId: Long) {
        val hospital = hospitalRepository.findByIdOrThrow(hospitalId) // 존재하는 병원인지 유효성 체크.
        val mike = mikeRepository.findByIdOrThrow(mikeId)
        accidentRepository.save(
            Accident(
                startTime = LocalDateTime.now(),
                endTime = LocalDateTime.now(),
                mike = mike,
                hospital = hospital
            )
        )
        sendClientMessage("/ws/topic/accident/hospitals/$hospitalId/audio", hospitalId, mike) // 클라이언트로 사고 발생 알림 보내기
    }

    private fun sendClientMessage(topicPath: String, hospitalId: Long, mike: Mike) {
        logInfo("Send WebSocket STOMP Audio Message To Path: $topicPath")
        messagingTemplate.convertAndSend(
            topicPath,
            AudioAccidentMessageDto(
                hospitalId = hospitalId,
                mike = MikeMessageDto(
                    id = mike.id!!,
                    content = mike.name
                ),
                startTime = LocalDateTime.now(),
                endTime = LocalDateTime.now()
            )
        )
    }
}
