import streamlit as st
import pandas as pd

# ===== Thông tin sản phẩm =====
product_info = {
    "Tên sản phẩm": "Máy tạo ẩm mini 250ml Portable",
    "Mô tả": "Máy tạo ẩm mini, giúp tăng độ ẩm không khí, khử mùi, có thể dùng trong xe hơi hoặc phòng nhỏ.",
    "Tổng điểm": 84,
    "Xếp loại": "✅ Nên triển khai kinh doanh",
    "Tóm tắt": "Máy tạo ẩm mini 250ml là sản phẩm có tiềm năng, dễ bán online, biên lợi nhuận ổn, tuy có cạnh tranh cao nhưng phù hợp nếu xây dựng thương hiệu tốt."
}

criteria = [
    ("Nhu cầu thị trường", 8),
    ("Mức độ cạnh tranh", 6),
    ("Biên lợi nhuận", 7),
    ("Tính độc đáo / USP", 6),
    ("Tính dễ bán online", 9),
    ("Tính lặp lại / upsell", 7),
    ("Rủi ro nhập hàng", 7),
    ("Khả năng mở rộng thương hiệu", 6),
    ("Feedback người dùng", 8),
    ("Pháp lý / Quảng cáo", 9),
    ("Rào cản gia nhập", 7),
    ("Tính bền vững", 4),
]

# ===== Layout =====
st.set_page_config(page_title="Dashboard Sản phẩm", layout="wide")

st.title("📦 Dashboard Đánh Giá Sản Phẩm")
st.markdown("---")

# ===== Thông tin cơ bản =====
st.header("🛍️ Thông tin sản phẩm")
st.write(f"**Tên sản phẩm:** {product_info['Tên sản phẩm']}")
st.write(f"**Mô tả:** {product_info['Mô tả']}")

# ===== Bảng tiêu chí =====
st.header("✅ Bảng tiêu chí đánh giá")
df = pd.DataFrame(criteria, columns=["Tiêu chí", "Điểm"])
st.dataframe(df, use_container_width=True)

# ===== Tổng kết =====
st.header("📊 Tổng điểm & Đánh giá chung")
st.metric(label="Tổng điểm", value=product_info["Tổng điểm"])
st.success(product_info["Xếp loại"])
st.markdown(f"**Tóm tắt:** {product_info['Tóm tắt']}")

# ===== Gợi ý chiến lược bán =====
st.header("📈 Chiến lược bán hàng đề xuất")

with st.expander("📍 Kênh bán hàng phù hợp"):
    st.markdown("""
- **TikTok:** Video ngắn, ASMR, review nhanh
- **Facebook:** Tạo cộng đồng, chạy ads văn phòng
- **Shopee/Lazada:** Scale đơn hàng theo khối lượng
- **Website riêng:** Nếu có thương hiệu mạnh
    """)

with st.expander("💡 Ý tưởng content"):
    st.markdown("""
- Review chi tiết sản phẩm
- Video mở hộp (unbox)
- Hướng dẫn sử dụng, bảo quản
- ASMR thư giãn
- Livestream + minigame
    """)

with st.expander("💰 Chiến lược giá & upsell"):
    st.markdown("""
- Giá cạnh tranh
- Combo: Máy + tinh dầu
- Upsell phụ kiện lọc gió, đèn mini
    """)

with st.expander("🎯 Khách hàng mục tiêu"):
    st.markdown("""
- **Giới tính:** Nữ (70%), Nam (30%)  
- **Độ tuổi:** 22 - 35  
- **Nghề nghiệp:** Dân văn phòng, sinh viên  
- **Vị trí:** Thành phố lớn  
- **Sở thích:** Làm đẹp, sức khỏe, không gian sống  
- **Hành vi:** Mua online, thích tiện ích nhỏ gọn
    """)

# ===== Footer =====
st.markdown("---")
st.caption("© 2025 - Dashboard đánh giá sản phẩm bằng Streamlit")

