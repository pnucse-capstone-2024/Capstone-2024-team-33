package com.effective.detector.hospital.domain.converter

import com.effective.detector.hospital.domain.AgeType
import jakarta.persistence.AttributeConverter
import jakarta.persistence.Converter

@Converter(autoApply = true)
class AgeConverter : AttributeConverter<AgeType, String> {

    override fun convertToDatabaseColumn(attribute: AgeType?): String? {
        return attribute?.value
    }

    override fun convertToEntityAttribute(dbData: String?): AgeType? {
        return dbData?.let { value ->
            AgeType.entries.find { it.value == value }
        }
    }
}
