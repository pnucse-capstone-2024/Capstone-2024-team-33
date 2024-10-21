package com.effective.detector.hospital.domain.converter

import com.effective.detector.hospital.domain.HospitalType
import jakarta.persistence.AttributeConverter
import jakarta.persistence.Converter

@Converter(autoApply = true)
class HospitalTypeConverter : AttributeConverter<HospitalType, String> {

    override fun convertToDatabaseColumn(attribute: HospitalType?): String? {
        return attribute?.value
    }

    override fun convertToEntityAttribute(dbData: String?): HospitalType? {
        return dbData?.let { value ->
            HospitalType.entries.find { it.value == value }
        }
    }
}
