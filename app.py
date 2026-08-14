import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from mappings import (
    MARITAL_STATUS, APPLICATION_MODE, COURSE, PREVIOUS_QUALIFICATION,
    NACIONALITY, QUALIFICATION_LEVEL, YES_NO, GENDER, ATTENDANCE,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "dropout_prediction_model.joblib")

MOTHERS_OCCUPATION_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134,
                            141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194]
FATHERS_OCCUPATION_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121,
                            122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154,
                            161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195]

FEATURE_ORDER = [
    "Marital_status", "Application_mode", "Application_order", "Course",
    "Daytime_evening_attendance", "Previous_qualification", "Previous_qualification_grade",
    "Nacionality", "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
    "Fathers_occupation", "Admission_grade", "Displaced", "Educational_special_needs",
    "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "Age_at_enrollment",
    "International", "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate", "Inflation_rate", "GDP",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def selectbox_mapped(label, mapping, codes, default_code, key, help_text=None):
    codes_sorted = sorted(codes, key=lambda c: mapping.get(c, str(c)))
    default_index = codes_sorted.index(default_code) if default_code in codes_sorted else 0
    choice = st.selectbox(
        label,
        options=codes_sorted,
        index=default_index,
        format_func=lambda c: mapping.get(c, f"Kode {c}"),
        key=key,
        help=help_text,
    )
    return choice


st.set_page_config(page_title="Prediksi Risiko Dropout Mahasiswa", page_icon="🎓", layout="wide")

st.title("🎓 Prediksi Risiko Dropout Mahasiswa — Jaya Jaya Institut")
st.markdown(
    """
Aplikasi ini membantu staf akademik memperkirakan **risiko dropout** seorang mahasiswa berdasarkan
data demografis, sosial-ekonomi, admisi, dan performa akademik semester 1 & 2, menggunakan model
**Logistic Regression** yang telah dilatih pada data historis mahasiswa Jaya Jaya Institut.

Isi form di bawah sesuai data mahasiswa, lalu klik **Prediksi Risiko Dropout**.
"""
)

if not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan di `{MODEL_PATH}`. Pastikan folder `model/` tersedia.")
    st.stop()

model = load_model()

with st.form("prediction_form"):
    st.subheader("1️⃣ Data Demografis")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = selectbox_mapped("Jenis Kelamin", GENDER, [0, 1], 0, "gender")
        marital_status = selectbox_mapped("Status Pernikahan", MARITAL_STATUS, list(MARITAL_STATUS.keys()), 1, "marital")
    with c2:
        age = st.number_input("Usia saat Pendaftaran (tahun)", min_value=17, max_value=70, value=20, step=1, key="age")
        nacionality = selectbox_mapped("Kewarganegaraan", NACIONALITY, list(NACIONALITY.keys()), 1, "nacionality")
    with c3:
        displaced = selectbox_mapped("Status Displaced (Pindah Domisili)", YES_NO, [0, 1], 0, "displaced")
        international = selectbox_mapped("Mahasiswa Internasional", YES_NO, [0, 1], 0, "international")

    with st.expander("Data Demografis Tambahan (Orang Tua & Kebutuhan Khusus)"):
        c4, c5 = st.columns(2)
        with c4:
            mothers_qualification = selectbox_mapped("Pendidikan Terakhir Ibu", QUALIFICATION_LEVEL, list(QUALIFICATION_LEVEL.keys()), 1, "mq")
            mothers_occupation = st.selectbox("Kode Pekerjaan Ibu", options=MOTHERS_OCCUPATION_CODES, index=0, key="mocc",
                                               help="Kode pekerjaan sesuai klasifikasi resmi dataset (tidak seluruhnya terdokumentasi secara publik).")
            educational_special_needs = selectbox_mapped("Kebutuhan Khusus dalam Pendidikan", YES_NO, [0, 1], 0, "esn")
        with c5:
            fathers_qualification = selectbox_mapped("Pendidikan Terakhir Ayah", QUALIFICATION_LEVEL, list(QUALIFICATION_LEVEL.keys()), 1, "fq")
            fathers_occupation = st.selectbox("Kode Pekerjaan Ayah", options=FATHERS_OCCUPATION_CODES, index=0, key="focc",
                                               help="Kode pekerjaan sesuai klasifikasi resmi dataset (tidak seluruhnya terdokumentasi secara publik).")

    st.subheader("2️⃣ Data Penerimaan (Admission)")
    c6, c7, c8 = st.columns(3)
    with c6:
        application_mode = selectbox_mapped("Jalur Pendaftaran", APPLICATION_MODE, list(APPLICATION_MODE.keys()), 1, "appmode")
        application_order = st.number_input("Urutan Pilihan Program Studi (0=pilihan pertama)", min_value=0, max_value=9, value=1, step=1, key="apporder")
    with c7:
        course = selectbox_mapped("Program Studi", COURSE, list(COURSE.keys()), 9119, "course")
        attendance = selectbox_mapped("Waktu Perkuliahan", ATTENDANCE, [0, 1], 1, "attendance")
    with c8:
        previous_qualification = selectbox_mapped("Pendidikan Sebelumnya", PREVIOUS_QUALIFICATION, list(PREVIOUS_QUALIFICATION.keys()), 1, "prevqual")
        previous_qualification_grade = st.number_input("Nilai Pendidikan Sebelumnya (0-200)", min_value=0.0, max_value=200.0, value=133.0, step=0.5, key="prevgrade")
    admission_grade = st.number_input("Nilai Admisi (0-200)", min_value=0.0, max_value=200.0, value=126.0, step=0.5, key="admgrade")

    st.subheader("3️⃣ Data Finansial")
    c9, c10, c11 = st.columns(3)
    with c9:
        debtor = selectbox_mapped("Memiliki Tunggakan Pembayaran (Debtor)?", YES_NO, [0, 1], 0, "debtor")
    with c10:
        tuition_paid = selectbox_mapped("Biaya Kuliah Lunas Tepat Waktu?", YES_NO, [0, 1], 1, "tuition")
    with c11:
        scholarship = selectbox_mapped("Penerima Beasiswa?", YES_NO, [0, 1], 0, "scholarship")

    st.subheader("4️⃣ Performa Akademik Semester 1")
    c12, c13, c14 = st.columns(3)
    with c12:
        cu1_credited = st.number_input("Jumlah Mata Kuliah Dikreditkan (Sem 1)", min_value=0, max_value=26, value=0, step=1, key="cu1c")
        cu1_enrolled = st.number_input("Jumlah Mata Kuliah Diambil (Sem 1)", min_value=0, max_value=26, value=6, step=1, key="cu1e")
    with c13:
        cu1_evaluations = st.number_input("Jumlah Evaluasi Mata Kuliah (Sem 1)", min_value=0, max_value=45, value=8, step=1, key="cu1ev")
        cu1_approved = st.number_input("Jumlah Mata Kuliah Lulus/Disetujui (Sem 1)", min_value=0, max_value=26, value=5, step=1, key="cu1a")
    with c14:
        cu1_grade = st.number_input("Nilai Rata-rata Semester 1 (0-20)", min_value=0.0, max_value=20.0, value=12.3, step=0.1, key="cu1g")
        cu1_no_eval = st.number_input("Jumlah Mata Kuliah Tanpa Evaluasi (Sem 1)", min_value=0, max_value=12, value=0, step=1, key="cu1ne")

    st.subheader("5️⃣ Performa Akademik Semester 2")
    c15, c16, c17 = st.columns(3)
    with c15:
        cu2_credited = st.number_input("Jumlah Mata Kuliah Dikreditkan (Sem 2)", min_value=0, max_value=23, value=0, step=1, key="cu2c")
        cu2_enrolled = st.number_input("Jumlah Mata Kuliah Diambil (Sem 2)", min_value=0, max_value=23, value=6, step=1, key="cu2e")
    with c16:
        cu2_evaluations = st.number_input("Jumlah Evaluasi Mata Kuliah (Sem 2)", min_value=0, max_value=33, value=8, step=1, key="cu2ev")
        cu2_approved = st.number_input("Jumlah Mata Kuliah Lulus/Disetujui (Sem 2)", min_value=0, max_value=20, value=5, step=1, key="cu2a")
    with c17:
        cu2_grade = st.number_input("Nilai Rata-rata Semester 2 (0-20)", min_value=0.0, max_value=20.0, value=12.2, step=0.1, key="cu2g")
        cu2_no_eval = st.number_input("Jumlah Mata Kuliah Tanpa Evaluasi (Sem 2)", min_value=0, max_value=12, value=0, step=1, key="cu2ne")

    with st.expander("Indikator Ekonomi Makro (opsional, gunakan nilai default jika tidak tahu)"):
        c18, c19, c20 = st.columns(3)
        with c18:
            unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=30.0, value=11.1, step=0.1, key="unemp")
        with c19:
            inflation_rate = st.number_input("Tingkat Inflasi (%)", min_value=-5.0, max_value=10.0, value=1.4, step=0.1, key="infl")
        with c20:
            gdp = st.number_input("GDP", min_value=-10.0, max_value=10.0, value=0.32, step=0.01, key="gdp")

    submitted = st.form_submit_button("🔍 Prediksi Risiko Dropout", use_container_width=True)

if submitted:
    # Validasi input sederhana: pastikan jumlah lulus tidak melebihi jumlah diambil
    errors = []
    if cu1_approved > cu1_enrolled:
        errors.append("Jumlah mata kuliah lulus semester 1 tidak boleh lebih besar dari jumlah mata kuliah yang diambil.")
    if cu2_approved > cu2_enrolled:
        errors.append("Jumlah mata kuliah lulus semester 2 tidak boleh lebih besar dari jumlah mata kuliah yang diambil.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        input_dict = {
            "Marital_status": marital_status,
            "Application_mode": application_mode,
            "Application_order": application_order,
            "Course": course,
            "Daytime_evening_attendance": attendance,
            "Previous_qualification": previous_qualification,
            "Previous_qualification_grade": previous_qualification_grade,
            "Nacionality": nacionality,
            "Mothers_qualification": mothers_qualification,
            "Fathers_qualification": fathers_qualification,
            "Mothers_occupation": mothers_occupation,
            "Fathers_occupation": fathers_occupation,
            "Admission_grade": admission_grade,
            "Displaced": displaced,
            "Educational_special_needs": educational_special_needs,
            "Debtor": debtor,
            "Tuition_fees_up_to_date": tuition_paid,
            "Gender": gender,
            "Scholarship_holder": scholarship,
            "Age_at_enrollment": age,
            "International": international,
            "Curricular_units_1st_sem_credited": cu1_credited,
            "Curricular_units_1st_sem_enrolled": cu1_enrolled,
            "Curricular_units_1st_sem_evaluations": cu1_evaluations,
            "Curricular_units_1st_sem_approved": cu1_approved,
            "Curricular_units_1st_sem_grade": cu1_grade,
            "Curricular_units_1st_sem_without_evaluations": cu1_no_eval,
            "Curricular_units_2nd_sem_credited": cu2_credited,
            "Curricular_units_2nd_sem_enrolled": cu2_enrolled,
            "Curricular_units_2nd_sem_evaluations": cu2_evaluations,
            "Curricular_units_2nd_sem_approved": cu2_approved,
            "Curricular_units_2nd_sem_grade": cu2_grade,
            "Curricular_units_2nd_sem_without_evaluations": cu2_no_eval,
            "Unemployment_rate": unemployment_rate,
            "Inflation_rate": inflation_rate,
            "GDP": gdp,
        }
        X_input = pd.DataFrame([[input_dict[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

        proba_dropout = float(model.predict_proba(X_input)[0, 1])
        prediction = int(proba_dropout >= 0.5)

        st.divider()
        st.subheader("📊 Hasil Prediksi")

        col_result, col_gauge = st.columns([1, 1])
        with col_result:
            if prediction == 1:
                st.error(f"### ⚠️ Risiko Dropout: TINGGI")
            else:
                st.success(f"### ✅ Risiko Dropout: RENDAH")
            st.metric("Probabilitas Risiko Dropout", f"{proba_dropout*100:.1f}%")

        with col_gauge:
            st.progress(min(max(proba_dropout, 0.0), 1.0))
            if proba_dropout >= 0.5:
                st.markdown(
                    f"Model memperkirakan probabilitas mahasiswa ini **dropout sebesar {proba_dropout*100:.1f}%**, "
                    "melebihi ambang batas 50%. Disarankan mahasiswa ini masuk daftar **prioritas monitoring/bimbingan akademik**."
                )
            else:
                st.markdown(
                    f"Model memperkirakan probabilitas mahasiswa ini **dropout sebesar {proba_dropout*100:.1f}%**, "
                    "di bawah ambang batas 50%. Risiko relatif rendah berdasarkan data yang dimasukkan, namun tetap disarankan pemantauan berkala."
                )

        st.info(
            "ℹ️ **Interpretasi:** performa akademik (jumlah mata kuliah yang disetujui/lulus) dan kondisi finansial "
            "(status pembayaran tuition, status tunggakan) merupakan faktor yang paling berkaitan dengan risiko dropout "
            "berdasarkan analisis model. Nilai probabilitas di atas dihitung langsung oleh model dan bukan estimasi manual."
        )

        st.warning(
            "⚠️ **Disclaimer:** hasil prediksi ini merupakan **alat bantu pendukung keputusan (decision support)** bagi "
            "dosen pembimbing akademik dan staf terkait, **bukan keputusan akademik otomatis**. Keputusan akhir terkait "
            "status atau intervensi mahasiswa tetap berada di tangan pihak berwenang di institusi, dengan mempertimbangkan "
            "konteks tambahan yang mungkin tidak tertangkap oleh model."
        )

st.divider()
st.caption(
    "Model: Logistic Regression (class_weight='balanced') · Dilatih pada data historis mahasiswa Jaya Jaya Institut "
    "(dataset resmi Dicoding: students_performance) · Prototype ini dibuat untuk keperluan proyek akhir Dicoding."
)
