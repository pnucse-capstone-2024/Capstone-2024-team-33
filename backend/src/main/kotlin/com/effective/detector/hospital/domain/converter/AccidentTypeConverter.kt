package com.effective.detector.hospital.domain.converter

import com.effective.detector.hospital.domain.AccidentType
import jakarta.persistence.AttributeConverter
import jakarta.persistence.Converter

@Converter(autoApply = true)
class AccidentTypeConverter : AttributeConverter<AccidentType, String> {

    override fun convertToDatabaseColumn(attribute: AccidentType?): String? {
        return attribute?.value
    }

    override fun convertToEntityAttribute(dbData: String?): AccidentType? {
        return dbData?.let { value ->
            AccidentType.entries.find { it.value == value }
        }
    }
}
