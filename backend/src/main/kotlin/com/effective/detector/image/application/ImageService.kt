package com.effective.detector.image.application

import com.effective.detector.common.error.BusinessError
import com.effective.detector.common.error.BusinessException
import com.effective.detector.common.util.findByIdOrThrow
import com.effective.detector.common.util.logError
import com.effective.detector.common.util.logInfo
import com.effective.detector.hospital.domain.*
import com.fasterxml.jackson.annotation.JsonCreator
import com.fasterxml.jackson.annotation.JsonProperty
import org.springframework.amqp.core.Message
import org.springframework.amqp.rabbit.core.RabbitTemplate
import org.springframework.beans.factory.annotation.Value
import org.springframework.messaging.simp.SimpMessagingTemplate
import org.springframework.stereotype.Service
import java.time.LocalDateTime
import java.util.*


@Service
class ImageService(
    private val rabbitTemplate: RabbitTemplate,
    private val cameraRepository: CameraRepository,
    private val messagingTemplate: SimpMessagingTemplate,
) {

    @Value("\${rabbitmq.ttl}")
    private val ttl: String? = null

    fun sendMessage(byteArray: ByteArray, queueName: String, hospitalId: Long, cameraId: Long) {
        logInfo("send to rabbitMq image file")
        try {
            rabbitTemplate.convertAndSend("", queueName, ImageMessageDto(byteArray)) { message: Message ->
                message.messageProperties.expiration = ttl
                message
            }
            sendClientImage(cameraId, byteArray)
        } catch (exception: Exception) {
            logError(exception.toString(), exception)
            throw BusinessException(BusinessError.RABBITMQ_CONNECTION_ERROR)
        }
    }

    private fun sendClientImage(cameraId: Long, readBytes: ByteArray) {
        val camera = cameraRepository.findByIdOrThrow(cameraId)
        val topicPath = "/ws/topic/image/hospitals/${camera.hospital.id}/cameras/$cameraId"
        logInfo("Send WebSocket STOMP Message To Path: $topicPath")
        val base64EncodedImage = Base64.getEncoder().encodeToString(readBytes)
        messagingTemplate.convertAndSend(
            topicPath,
            com.effective.detector.common.handler.ImageMessageDto(
                encodedImage = base64EncodedImage
            )
        )
    }
}

data class ImageMessageDto @JsonCreator constructor(
    @JsonProperty("bytes") val bytes: ByteArray,
    @JsonProperty("time") val time: String = LocalDateTime.now().toString(),
)

data class AccidentMessageDto @JsonCreator constructor(
    @JsonProperty("hospitalId") val hospitalId: Long,
    @JsonProperty("camera") val camera: CameraMessageDto,
    @JsonProperty("startTime") val startTime: LocalDateTime,
    @JsonProperty("endTime") val endTime: LocalDateTime,
)

data class AudioAccidentMessageDto @JsonCreator constructor(
    @JsonProperty("hospitalId") val hospitalId: Long,
    @JsonProperty("mike") val mike: MikeMessageDto,
    @JsonProperty("startTime") val startTime: LocalDateTime,
    @JsonProperty("endTime") val endTime: LocalDateTime,
)

data class CameraMessageDto @JsonCreator constructor(
    @JsonProperty("id") val id: Long,
    @JsonProperty("content") val content: String,
)

data class MikeMessageDto @JsonCreator constructor(
    @JsonProperty("id") val id: Long,
    @JsonProperty("content") val content: String,
)
