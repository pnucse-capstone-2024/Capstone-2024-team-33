package com.effective.detector.common.config

import com.amazonaws.auth.AWSCredentials
import com.amazonaws.auth.AWSStaticCredentialsProvider
import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.regions.Regions.AP_NORTHEAST_2
import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class S3Config {

    @Value("\${aws.access-key}")
    private val accessKey: String? = null

    @Value("\${aws.secret-key}")
    private val secretKey: String? = null

    @Bean
    fun amazonS3Client(): AmazonS3 {
        val credentials: AWSCredentials = BasicAWSCredentials(accessKey, secretKey)
        return AmazonS3ClientBuilder
            .standard()
            .withCredentials(AWSStaticCredentialsProvider(credentials))
            .withRegion(AP_NORTHEAST_2)
            .build()
    }
}
