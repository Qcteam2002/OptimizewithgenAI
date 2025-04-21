import json

seo_product_prompt = {
    "task": "Viết tiêu đề và mô tả chuẩn SEO, tăng chuyển đổi và tạo cảm xúc mua hàng.",
    "goal": "Tạo nội dung thu hút, tối ưu SEO, tăng tỷ lệ mua ngay.",
    "targetaudience": "[chèn đặc điểm khách hàng tiềm năng]",
    "toneandvoice": "Cao cấp → chuyên nghiệp. Thời trang → sáng tạo, cảm xúc. Gia dụng → thân thiện. Ngôn ngữ ngắn gọn, rõ ràng, chia đoạn dễ đọc."
}

def generate_prompt(title, description, library_data, featured_media, image):
    """
    Tạo prompt cho OpenAI dựa trên thông tin sản phẩm
    
    Args:
        title (str): Tiêu đề sản phẩm
        description (str): Mô tả sản phẩm
        library_data (dict): Dữ liệu thư viện
        featured_media (str): URL media nổi bật
        image (str or list): URL hình ảnh hoặc danh sách URL hình ảnh
        
    Returns:
        str: JSON string chứa prompt
    """
    # Xử lý image thành list
    if isinstance(image, list):
        image_list = [img.get('url') if isinstance(img, dict) else img for img in image]
    else:
        image_list = [image] if image else ["No image available"]
    
    prompt_data = {
        "language": "English",
        "seoProductPrompt": seo_product_prompt,
        "referenceLibrary": library_data,
        "input": {
            "title": title,
            "description": description,
            "featuredMedia": featured_media or "No featured media available.",
            "imageList": image_list
        },
        "jsonOutput": {
            "requirement": "**BẮT BUỘC** chỉ trả về JSON hợp lệ theo đúng format này. Không thêm bất kỳ văn bản nào ngoài JSON.",
            "outputExample": {
                "optimizedTitle": "<50-65 word Optimized Product Title>",
                "optimizedDescription": "<100-200 word optimized product description with proper line breaks>",
                "gridView": [
                    {
                        "title": "Highlighting the Main USP - [Product's Unique Feature]",
                        "description": "Mô tả tính năng nổi bật nhất và lợi ích độc quyền sản phẩm mang lại...",
                        "image": "Chọn 1 ảnh trong imageList phù hợp nhất với tính năng này"
                    },
                    {
                        "title": "Advanced Technology - [Key Functional Feature]",
                        "description": "Giới thiệu công nghệ/ứng dụng sản phẩm trong thực tế, nhấn mạnh tiện ích...",
                        "image": "Chọn 1 ảnh thể hiện công năng/ứng dụng tính năng này trong đời sống"
                    },
                    {
                        "title": "Proven Reliability - [Distinctive Feature]",
                        "description": "Nêu bật đặc điểm độc đáo và bằng chứng về hiệu quả hoặc độ tin cậy...",
                        "image": "Chọn 1 ảnh thể hiện đặc điểm riêng biệt này rõ ràng nhất từ imageList"
                    }
                ]
            },
            "rules": {
                "optimizedDescription": "phải dùng '\\n' để xuống dòng rõ ràng.",
                "gridView": {
                    "imageSelection": "Dựa vào title & description, xác định keyword chính của mỗi tính năng. So sánh với imageList, chọn ra ảnh minh họa trực quan và liên quan nhất. Trả về đúng 1 URL duy nhất mỗi tính năng."
                }
            }
        }
    }
    
    return json.dumps(prompt_data, ensure_ascii=False) 