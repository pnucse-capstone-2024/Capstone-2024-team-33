package com.effective.detector.hospital.domain

enum class AgeType(
    val id: Long,
    val value: String,
) {
    AGE_10(1, "10대"),
    AGE_20(2, "20대"),
    AGE_30(3, "30대"),
    AGE_40(4, "40대"),
    AGE_50(5, "50대"),
    AGE_60(6, "60대"),
    AGE_70(7, "70대"),
    AGE_80(8, "80대"),
    AGE_NONE(9, "미상"),
    ;

    companion object {
        fun from(id: Long): AgeType {
            return entries.first { it.id == id }
        }
    }
}
