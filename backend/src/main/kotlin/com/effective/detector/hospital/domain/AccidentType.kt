package com.effective.detector.hospital.domain

enum class AccidentType(
    val id: Long,
    val value: String,
) {
    SLIPPIING(1, "미끄러짐"),
    FIGHTING(2, "환자 간 다툼"),
    POOR_FACILITIES(3, "시설 부실"),
    CARELESSNESS(4, "의료진 부주의"),
    ETC(5, "기타"),
    MALFUNCTION(6, "오작동"),
    ;

    companion object {
        fun from(id: Long): AccidentType {
            return entries.first { it.id == id }
        }
    }
}
