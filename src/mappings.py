"""Mapping kode numerik ke label manusiawi berdasarkan dokumentasi resmi dataset
students_performance (Dicoding) / UCI Machine Learning Repository (Realinho et al., 2021).
"""

MARITAL_STATUS = {
    1: "Belum Menikah (Single)",
    2: "Menikah (Married)",
    3: "Duda/Janda (Widower)",
    4: "Bercerai (Divorced)",
    5: "Kumpul Kebo (Facto Union)",
    6: "Berpisah Secara Hukum (Legally Separated)",
}

APPLICATION_MODE = {
    1: "Fase 1 - Kontingen Umum",
    2: "Ordinance No. 612/93",
    5: "Fase 1 - Kontingen Khusus (Pulau Azores)",
    7: "Pemegang Kualifikasi Pendidikan Tinggi Lain",
    10: "Ordinance No. 854-B/99",
    15: "Mahasiswa Internasional (Sarjana)",
    16: "Fase 1 - Kontingen Khusus (Pulau Madeira)",
    17: "Fase 2 - Kontingen Umum",
    18: "Fase 3 - Kontingen Umum",
    26: "Ordinance No. 533-A/99, item b2 (Rencana Berbeda)",
    27: "Ordinance No. 533-A/99, item b3 (Institusi Lain)",
    39: "Usia Diatas 23 Tahun",
    42: "Pindah (Transfer)",
    43: "Perubahan Program Studi",
    44: "Pemegang Diploma Spesialisasi Teknologi",
    51: "Pindah Institusi/Program Studi",
    53: "Pemegang Diploma Siklus Singkat",
    57: "Pindah Institusi/Program Studi (Internasional)",
}

COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (Kelas Malam)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Kelas Malam)",
}

PREVIOUS_QUALIFICATION = {
    1: "Pendidikan Menengah (SMA)",
    2: "Pendidikan Tinggi - Bachelor",
    3: "Pendidikan Tinggi - Degree",
    4: "Pendidikan Tinggi - Master",
    5: "Pendidikan Tinggi - Doctorate",
    6: "Frequency of Higher Education",
    9: "Kelas 12 - Tidak Selesai",
    10: "Kelas 11 - Tidak Selesai",
    12: "Lainnya - Kelas 11",
    14: "Kelas 10",
    15: "Kelas 10 - Tidak Selesai",
    19: "Pendidikan Dasar Siklus 3 (Kelas 9/10/11)",
    38: "Pendidikan Dasar Siklus 2 (Kelas 6/7/8)",
    39: "Technological Specialization Course",
    40: "Pendidikan Tinggi - Degree (Siklus 1)",
    42: "Professional Higher Technical Course",
    43: "Pendidikan Tinggi - Master (Siklus 2)",
}

NACIONALITY = {
    1: "Portugis", 2: "Jerman", 6: "Spanyol", 11: "Italia", 13: "Belanda",
    14: "Inggris", 17: "Lithuania", 21: "Angola", 22: "Cape Verde", 24: "Guinea",
    25: "Mozambik", 26: "Santomean", 32: "Turki", 41: "Brazil", 62: "Rumania",
    100: "Moldova", 101: "Meksiko", 103: "Ukraina", 105: "Rusia", 108: "Kuba", 109: "Kolombia",
}

QUALIFICATION_LEVEL = {
    1: "Pendidikan Menengah (Kelas 12)",
    2: "Pendidikan Tinggi - Bachelor",
    3: "Pendidikan Tinggi - Degree",
    4: "Pendidikan Tinggi - Master",
    5: "Pendidikan Tinggi - Doctorate",
    6: "Frequency of Higher Education",
    9: "Kelas 12 - Tidak Selesai",
    10: "Kelas 11 - Tidak Selesai",
    11: "Kelas 7 (Lama)",
    12: "Lainnya - Kelas 11",
    14: "Kelas 10",
    18: "Kursus Perdagangan Umum",
    19: "Pendidikan Dasar Siklus 3 (Kelas 9/10/11)",
    22: "Kursus Teknis-Profesional",
    26: "Kelas 7",
    27: "Siklus 2 Sekolah Menengah Umum",
    29: "Kelas 9 - Tidak Selesai",
    30: "Kelas 8",
    34: "Tidak Diketahui",
    35: "Tidak Bisa Baca Tulis",
    36: "Bisa Membaca tanpa Kelas 4",
    37: "Pendidikan Dasar Siklus 1 (Kelas 4/5)",
    38: "Pendidikan Dasar Siklus 2 (Kelas 6/7/8)",
    39: "Technological Specialization Course",
    40: "Pendidikan Tinggi - Degree (Siklus 1)",
    41: "Specialized Higher Studies Course",
    42: "Professional Higher Technical Course",
    43: "Pendidikan Tinggi - Master (Siklus 2)",
    44: "Pendidikan Tinggi - Doctorate (Siklus 3)",
}

YES_NO = {0: "Tidak", 1: "Ya"}
GENDER = {1: "Laki-laki", 0: "Perempuan"}
ATTENDANCE = {1: "Siang (Daytime)", 0: "Malam (Evening)"}


def label_for(mapping: dict, code: int) -> str:
    return mapping.get(code, f"Kode {code}")


def options_from(mapping: dict, present_codes: list) -> list:
    return sorted(present_codes, key=lambda c: mapping.get(c, ""))
