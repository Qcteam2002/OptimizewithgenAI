from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SAVE_PRODUCT_API = f"{os.getenv('API_BASE_URL')}/api/save-optimized-product"

# Load library.json
try:
    with open('./src/library.json', 'r', encoding='utf-8') as f:
        library_data = json.load(f)
        print("📚 Thư viện dữ liệu đã load thành công!")
except Exception as e:
    library_data = {}
    print("❌ Lỗi khi đọc thư viện dữ liệu:", e)

def generate_prompt(title, description, library, featured_media, image_list):
    return {
        "language": "English",
        "seoProductPrompt": {
            "task": "Viết tiêu đề và mô tả chuẩn SEO, tăng chuyển đổi và tạo cảm xúc mua hàng.",
            "goal": "Tạo nội dung thu hút, tối ưu SEO, tăng tỷ lệ mua ngay.",
            "targetaudience": "[chèn đặc điểm khách hàng tiềm năng]",
            "toneandvoice": "Cao cấp → chuyên nghiệp. Thời trang → sáng tạo, cảm xúc. Gia dụng → thân thiện. Ngôn ngữ ngắn gọn, rõ ràng, chia đoạn dễ đọc."
        },
        "referenceLibrary": library,
        "input": {
            "title": title,
            "description": description,
            "featuredMedia": featured_media,
            "imageList": [img.get("url") for img in image_list]
        },
        "jsonOutput": {
            "requirement": "**BẮT BUỘC** chỉ trả về JSON hợp lệ theo đúng format này. Không thêm bất kỳ văn bản nào ngoài JSON.",
            "outputExample": {
                "optimizedTitle": "Best Selling Stainless Steel Necklace",
                "optimizedDescription": "This necklace is crafted from high-grade stainless steel...\n...",
                "gridView": []
            },
            "rules": {
                "optimizedDescription": "phải dùng '\\n' để xuống dòng rõ ràng.",
                "gridView": {
                    "imageSelection": "Chọn ảnh minh hoạ trực quan, liên quan nhất."
                }
            }
        }
    }

@app.route("/api/openai", methods=["POST"])
def optimize():
    try:
        data = request.json
        id = data["id"]
        title = data["title"]
        description = data["description"]
        featured_media = data.get("featuredMedia", "")
        image_list = data.get("image", [])

        prompt = generate_prompt(title, description, library_data, featured_media, image_list)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia content SEO."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ]
        )

        content = response.choices[0].message.content.strip().strip("```json").strip("```")
        ai_data = json.loads(content)

        save_response = requests.post(SAVE_PRODUCT_API, json={
            "id": id,
            "optimizedTitle": ai_data.get("optimizedTitle"),
            "optimizedDescription": ai_data.get("optimizedDescription"),
            "gridView": ai_data.get("gridView")
        })

        return jsonify({
            "success": True,
            "data": ai_data,
            "dbResult": save_response.json()
        })

    except Exception as e:
        print("❌ Lỗi khi gọi OpenAI:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/openai/reviews", methods=["POST"])
def generate_reviews():
    try:
        data = request.json
        id = data["id"]
        title = data["title"]
        description = data["description"]
        featured_media = data["featuredMedia"]

        prompt = f"""
Trả lời bằng "Tiếnh Anh"
Bạn là AI chuyên viết đánh giá tổng quan cho sản phẩm eCommerce.

👉 Nhiệm vụ:
1. Dựa trên {title}, {description}, {featured_media}
2. Trả về đúng mảng JSON có 3 đánh giá như sau:
[
  {{
    "title": "Chất liệu dây",
    "comment": "Dây thép không gỉ, đeo êm tay, không dị ứng.",
    "star": "AI-generated review based on product details from multiple sources."
  }}
]
❗ Trả đúng JSON, không thêm markdown hoặc text ngoài.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là khách hàng ẩn danh để lại review sản phẩm."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip().strip("```json").strip("```")
        reviews = json.loads(content)

        save_response = requests.post(SAVE_PRODUCT_API, json={
            "id": id,
            "aiReviews": reviews
        })

        return jsonify({"reviews": reviews})

    except Exception as e:
        print("❌ Lỗi tạo review:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5004)
