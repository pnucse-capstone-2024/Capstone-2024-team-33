package com.effective.detector.hospital.domain

enum class HospitalType(
    val value: String,
) {
    HOSPITAL("병원"),
    OLD_NURSING_HOSPITAL("요양병원(노인병원)"),
    NURSING_HOSPITAL("요양병원(일반요양병원)"),
    DISABLED_NURSING_HOSPITAL("요양병원(장애인의료재활시설)"),
    MENTAL_HOSPITAL("정신병원"),
    GENERAL_HOSPITAL("종합병원"),
    DENTISTRY_HOSPITAL("치과병원"),
    ORIENTAL_MEDICINE_HOSPITAL("한방병원"),
    ;
}
