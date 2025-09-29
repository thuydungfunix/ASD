# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pickle

# Tải mô hình và scaler
model = pickle.load(open("logistic_asd_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Sàng Lọc Tự Kỷ", page_icon="🧠", layout="centered")
st.title("🔎 Ứng dụng Sàng Lọc Tự Kỷ (ASD)")

# Giao diện nhập thông tin
def nhap_du_lieu():
    st.header("📝 Vui lòng trả lời các câu hỏi sau:")

    # 10 câu hỏi A1 đến A10
    a1 = st.selectbox("1. Có khi nào người được đánh giá tránh giao tiếp bằng mắt?", ["Không", "Có"])
    a2 = st.selectbox("2. Người đó có thích chơi một mình?", ["Không", "Có"])
    a3 = st.selectbox("3. Người đó có hay lặp lại từ/ngôn ngữ không?", ["Không", "Có"])
    a4 = st.selectbox("4. Người đó có khó khăn khi hiểu cảm xúc người khác?", ["Không", "Có"])
    a5 = st.selectbox("5. Có khi nào người đó không phản hồi khi được gọi tên?", ["Không", "Có"])
    a6 = st.selectbox("6. Người đó có nhạy cảm với âm thanh không?", ["Không", "Có"])
    a7 = st.selectbox("7. Có khi nào người đó không chia sẻ hứng thú hoặc thành tích với người khác?", ["Không", "Có"])
    a8 = st.selectbox("8. Người đó có hành vi lặp đi lặp lại không?", ["Không", "Có"])
    a9 = st.selectbox("9. Người đó có gặp khó khăn khi thay đổi thói quen hoặc môi trường không?", ["Không", "Có"])
    a10 = st.selectbox("10. Có khi nào người đó không hiểu các quy tắc xã hội cơ bản không?", ["Không", "Có"])

    # Chuyển đổi thành số
    scores = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
    scores = [1 if s == "Có" else 0 for s in scores]

    # Các đặc trưng khác
    used_app_before = st.selectbox("11. Bạn đã từng sử dụng ứng dụng này chưa?", ["Chưa", "Rồi"])
    jundice = st.selectbox("12. Trẻ có bị vàng da sau sinh không?", ["Không", "Có"])
    austim = st.selectbox("13. Gia đình bạn có người từng bị tự kỷ không?", ["Không", "Có"])
    relation = st.selectbox("14. Bạn là ai đối với người được đánh giá?", ["Bố/mẹ", "Bản thân", "Người thân khác", "Khác"])
    age_desc = st.selectbox("15. Nhóm tuổi của người được đánh giá?", ["Dưới 18 tuổi", "Từ 18 tuổi trở lên"])

    # Mã hóa
    used_app_before = 1 if used_app_before == "Rồi" else 0
    jundice = 1 if jundice == "Có" else 0
    austim = 1 if austim == "Có" else 0
    relation_map = {"Bố/mẹ": 0, "Bản thân": 1, "Người thân khác": 2, "Khác": 3}
    relation = relation_map[relation]
    age_desc = 0 if age_desc == "Dưới 18 tuổi" else 1

    features = scores + [used_app_before, jundice, relation, austim, age_desc]
    return np.array([features])

# Dự đoán

def du_doan(features):
    scaled = scaler.transform(features)
    proba = model.predict_proba(scaled)[0][1]
    pred = int(proba >= 0.5)
    return pred, proba

# Chạy ứng dụng
input_data = nhap_du_lieu()
if st.button("📊 Dự đoán khả năng tự kỷ"):
    pred, proba = du_doan(input_data)

    st.subheader("🔍 Kết quả sàng lọc:")
    st.write(f"👉 Xác suất mắc tự kỷ (ASD): **{proba:.2f}**")

    st.markdown("### 🧭 Gợi ý hành động tiếp theo:")
    if pred == 1:
        st.info("""
        🔹 Hãy liên hệ chuyên gia tâm lý hoặc cơ sở y tế để được tư vấn kỹ lưỡng hơn.
        🔹 Ghi chép lại các biểu hiện thường gặp trong cuộc sống hàng ngày.
        🔹 Có thể tham khảo các tài liệu về ASD từ WHO, CDC hoặc các trung tâm hỗ trợ trong nước.
        """)
    else:
        st.success("""
        ✅ Bạn có thể yên tâm ở thời điểm hiện tại.
        🔹 Việc phát triển kỹ năng xã hội và cảm xúc vẫn ổn.
        🔹 Nếu còn băn khoăn, bạn có thể trao đổi thêm với chuyên gia.
        """)



