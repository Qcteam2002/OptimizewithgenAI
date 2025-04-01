from flask import Flask, render_template, request, jsonify, session, send_file, Response
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cv2
import numpy as np
import time
from datetime import datetime
import json
import tempfile
from werkzeug.utils import secure_filename
import requests
import subprocess

app = Flask(__name__)
app.secret_key = os.urandom(24)
client = genai.Client(api_key="AIzaSyAh_9Ku-QjqJ7o-gEGUvsK8dCNyygfD-q8")

# Tạo thư mục output nếu chưa tồn tại
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Global variable to store recording progress
recording_progress = 0

# Global dictionary to store video analysis results
video_analysis_results = {}

# Global variable for store analytics
next_store_analytics_id = 1
store_analytics = {}

# File để lưu trữ kết quả phân tích
ANALYTICS_FILE = 'store_analytics.json'

# Thêm biến và file để lưu trữ kết quả optimize image
OPTIMIZE_IMAGE_FILE = 'optimize_image_data.json'
optimize_image_data = {
    'next_id': 1,
    'results': {}
}

# Thêm các biến cấu hình
UPLOAD_FOLDER = 'static/uploads/videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Tạo thư mục upload nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Thêm biến để lưu thông tin video đã tải lên
UPLOADED_VIDEOS = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def record_webpage(url):
    global recording_progress
    try:
        print("\n[RECORD-WEBPAGE] Starting webpage recording process...")
        print(f"[RECORD-WEBPAGE] URL to record: {url}")
        
        recording_progress = 0
        # Cấu hình Chrome options
        print("[RECORD-WEBPAGE] Configuring Chrome options...")
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-popup-blocking')
        chrome_options.add_argument('--disable-extensions')  # Tắt extensions
        chrome_options.add_argument('--disable-javascript')  # Tắt JavaScript nếu không cần thiết
        chrome_options.add_argument('--disable-images')  # Tắt tải hình ảnh

        # Khởi động trình duyệt
        print("[RECORD-WEBPAGE] Starting Chrome browser...")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Đợi trang web tải xong (giảm thời gian chờ)
        print("[RECORD-WEBPAGE] Waiting for page to load (3 seconds)...")
        time.sleep(3)

        # Inject JS để loại bỏ popup
        remove_popup_script = """
        (() => {
        window.alert = function () {};
        window.confirm = function () { return true; };

        const keywords = ['popup', 'modal', 'overlay', 'dialog', 'klaviyo', 'subscribe'];
        keywords.forEach(k => {
            document.querySelectorAll(`[class*="${k}"], [id*="${k}"]`).forEach(e => e.remove());
        });

        document.querySelectorAll('[role="dialog"]').forEach(e => e.remove());
        })();
        """
        driver.execute_script(remove_popup_script)
        print("[RECORD-WEBPAGE] Popups removed via JS injection.")
        
        # Lấy kích thước trang
        print("[RECORD-WEBPAGE] Getting page dimensions...")
        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")
        print(f"[RECORD-WEBPAGE] Total height: {total_height}, Viewport height: {viewport_height}")
        
        # Tạo video writer với FPS cao hơn 
        print("[RECORD-WEBPAGE] Creating video writer...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f'recording_{timestamp}.avi')
        print(f"[RECORD-WEBPAGE] Video will be saved to: {output_path}")
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, 1.5, (1920, 1080))  # Tăng FPS lên 3.0 để video chạy nhanh hơn nhiều
        
        # Scroll và record
        print("\n[RECORD-WEBPAGE] Starting video recording process...")
        print("[RECORD-WEBPAGE] This process will:")
        print("1. Scroll through the webpage")
        print("2. Capture screenshots at each position")
        print("3. Convert screenshots to video frames")
        print("4. Save frames to video file")
        print("\n[RECORD-WEBPAGE] Recording progress:")
        
        current_position = 0
        scroll_step = 200  # Tăng bước scroll lên 200px
        frame_count = 0
        last_frame_time = time.time()
        
        while current_position < total_height:
            # Update progress
            progress = int((current_position / total_height) * 100)
            recording_progress = min(progress, 99)
            print(f"[RECORD-WEBPAGE] Current progress: {recording_progress}%")
            
            # Scroll mượt xuống
            for pos in range(current_position, min(current_position + viewport_height, total_height), scroll_step):
                driver.execute_script(f"window.scrollTo({{top: {pos}, behavior: 'auto'}});")
                time.sleep(0.1)  # Giảm thời gian chờ xuống 0.1 giây
                
                # Chụp screenshot và xử lý frame
                screenshot = driver.get_screenshot_as_png()
                image = Image.open(BytesIO(screenshot))
                image = image.convert('RGB')
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (1920, 1080))
                
                # Thêm frame vào video - không lặp lại frame
                out.write(frame)
                frame_count += 1
                
                # In log mỗi 5 frame
                if frame_count % 5 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - last_frame_time
                    print(f"[RECORD-WEBPAGE] Processed {frame_count} frames in {elapsed_time:.1f} seconds")
                    last_frame_time = current_time
            
            time.sleep(0.1)  # Giảm thời gian chờ giữa các lần scroll
            current_position += viewport_height
            
            if current_position >= total_height:
                break
        
        # Set progress to 100% when complete
        recording_progress = 100
        print("\n[RECORD-WEBPAGE] Recording completed!")
        print(f"[RECORD-WEBPAGE] Total frames recorded: {frame_count}")
        
        # Đóng video writer và trình duyệt
        print("[RECORD-WEBPAGE] Closing video writer and browser...")
        out.release()
        driver.quit()
        
        print(f"[RECORD-WEBPAGE] Video successfully saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[RECORD-WEBPAGE] ERROR: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        if 'out' in locals():
            out.release()
        return None

def generate_image_from_prompt(prompt, image1_data=None, image2_data=None):
    try:
        contents = [prompt]
        if image1_data:
            contents.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=image1_data)))
        if image2_data:
            contents.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=image2_data)))

        print("Prompt:", prompt)
        if image1_data:
            print("Image 1 uploaded")
        if image2_data:
            print("Image 2 uploaded")

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            ),
        )

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.mime_type == "image/png":
                    return part.inline_data.data
        return None
    except Exception as e:
        print("Error:", e)
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'clear_conversation' in request.form:
            session.pop('conversation', None)
            return jsonify({'message': 'Conversation cleared'})

        prompt = request.form['prompt']
        image1_data = request.files['image1'].read() if 'image1' in request.files else None
        image2_data = request.files['image2'].read() if 'image2' in request.files else None

        image_result = generate_image_from_prompt(prompt, image1_data, image2_data)
        text_result = None

        if image_result:
            image_base64 = base64.b64encode(image_result).decode('utf-8')
            return jsonify({'image': image_base64, 'text': None})
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["Text"]
                ),
            )
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.text:
                        text_result = part.text
            return jsonify({'image': None, 'text': text_result})

    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
            
        # Bắt đầu ghi lại trang web
        output_path = record_webpage(url)
        
        if output_path:
            return jsonify({
                'success': True,
                'message': 'Recording completed',
                'video_path': output_path
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to record webpage'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/record_progress')
def get_record_progress():
    def generate():
        global recording_progress
        while recording_progress < 100:
            yield f"data: {{'percentage': {recording_progress}}}\n\n"
            time.sleep(1)
        yield f"data: {{'percentage': 100}}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/recordings', methods=['GET'])
def get_recordings():
    try:
        recordings = []
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith('.mp4') or filename.endswith('.avi'):
                file_path = os.path.join(OUTPUT_DIR, filename)
                creation_time = os.path.getmtime(file_path)
                creation_datetime = datetime.fromtimestamp(creation_time)
                formatted_time = creation_datetime.strftime("%d/%m/%Y %H:%M:%S")
                
                recordings.append({
                    'name': filename,
                    'path': f'/output/{filename}',
                    'has_analysis': filename in video_analysis_results,
                    'creation_time': formatted_time,
                    'timestamp': creation_time
                })
        
        # Sắp xếp theo thời gian tạo, mới nhất đầu tiên
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'recordings': recordings
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/output/<path:filename>')
def serve_video(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename))

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        
        if not video_path:
            return jsonify({
                'success': False,
                'message': 'Video path is required'
            }), 400
            
        # Get filename from path
        filename = os.path.basename(video_path)
            
        # Check if analysis already exists
        if filename in video_analysis_results:
            print(f"[DEBUG] Using cached analysis for {filename}")
            return jsonify({
                'success': True,
                'analysis': video_analysis_results[filename]
            })
            
        # Chuyển đổi đường dẫn tương đối thành tuyệt đối
        absolute_video_path = video_path
        if video_path.startswith('/output/'):
            absolute_video_path = os.path.join(os.getcwd(), 'output', filename)
            
        print(f"[DEBUG] Analyzing video at absolute path: {absolute_video_path}")
        
        if not os.path.exists(absolute_video_path):
            return jsonify({
                'success': False,
                'message': f'Video file not found: {absolute_video_path}'
            }), 404
        
        # Phân tích video sử dụng Vertex AI Gemini
        print(f"[DEBUG] Using Vertex AI Gemini to analyze video")
        
        try:
            # Xác định MIME type dựa trên phần mở rộng của file
            mime_type = "video/mp4"
            if absolute_video_path.lower().endswith('.avi'):
                mime_type = "video/mp4"  # Sử dụng mp4 vì nhiều API hỗ trợ tốt hơn
                
            # Để sử dụng được với Vertex AI, chúng ta cần tạo file tạm thời
            # hoặc upload file lên GCS. Đây chúng ta sẽ sử dụng cách đơn giản hơn với file trực tiếp
            
            # Tạo prompt chi tiết cho phân tích
            prompt = """Hãy phân tích trang web bán sản phẩm này và cung cấp thông tin sau:
            1. Sản phẩm chính được bán là gì?
            2. Đối tượng khách hàng mục tiêu mà trang web hướng đến là ai?
            3. Trải nghiệm người dùng trên trang web này như thế nào? (Điều hướng, bố cục, tốc độ tải ước tính, tính thân thiện trên thiết bị di động)
            4. Thông tin về sản phẩm (mô tả, hình ảnh, giá cả) được trình bày như thế nào?
            5. Chính sách bán hàng và vận chuyển có rõ ràng và dễ hiểu không?
            6. Mức độ tin cậy của trang web (dựa trên các yếu tố có thể thấy trên trang).
            7. Đề xuất cải thiện để tăng doanh số và trải nghiệm người dùng (tối thiểu 3 đề xuất cụ thể)"""
            
            # Sử dụng phương pháp trích xuất frame do Vertex AI không hỗ trợ trực tiếp
            # đọc file video local mà không upload lên GCS
            
            # Trích xuất frames từ video
            cap = cv2.VideoCapture(absolute_video_path)
            if not cap.isOpened():
                return jsonify({
                    'success': False,
                    'message': f'Failed to open video: {absolute_video_path}'
                }), 500
                
            # Đếm số frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"[DEBUG] Video info: {total_frames} frames, {fps} fps, {duration:.2f} seconds")
            
            # Trích xuất 3 frame đại diện
            frames = []
            frame_indices = []
            
            if total_frames > 0:
                # Chọn 3 frame: đầu (10%), giữa (50%), cuối (90%)
                frame_indices = [int(total_frames * 0.1), int(total_frames * 0.5), int(total_frames * 0.9)]
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Giảm kích thước frame để giảm dung lượng
                        frame = cv2.resize(frame, (800, 450))
                        # Chuyển đổi frame thành file
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frames.append(buffer)
            
            cap.release()
            
            if not frames:
                return jsonify({
                    'success': False,
                    'message': 'No frames could be extracted from video'
                }), 500
            
            print(f"[DEBUG] Extracted {len(frames)} frames for analysis")
            
            # Initialize Vertex AI client
            client_options = {"api_endpoint": f"{GCP_LOCATION}-aiplatform.googleapis.com"}
            
            try:
                vertex_client = aiplatform.gapic.GenerativeModelServiceClient(client_options=client_options)
                model = generative_models.GenerativeModel(model_name="gemini-1.5-pro", client=vertex_client)
                
                # Tạo nội dung với frames
                contents = [generative_models.TextPart(text=prompt)]
                
                # Thêm frames vào nội dung
                for i, frame_buffer in enumerate(frames):
                    with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
                        temp_file.write(frame_buffer)
                        temp_file.flush()
                        
                        image_part = generative_models.Image.load_from_file(temp_file.name)
                        contents.append(image_part)
                
                # Gửi request đến Vertex AI
                response = model.generate_content(contents=contents)
                
                # Lấy kết quả phân tích
                analysis_result = response.text
                
                print(f"[DEBUG] Analysis result length: {len(analysis_result)} characters")
                
                # Store analysis result
                video_analysis_results[filename] = analysis_result
                
                return jsonify({
                    'success': True,
                    'analysis': analysis_result
                })
                
            except Exception as vertex_error:
                print(f"[DEBUG] Error with Vertex AI: {str(vertex_error)}")
                
                # Fallback to standard Gemini API if Vertex AI fails
                print("[DEBUG] Falling back to standard Gemini API")
                
                # Convert frames to base64
                frame_base64 = []
                for frame_buffer in frames:
                    frame_base64.append(base64.b64encode(frame_buffer).decode('utf-8'))
                
                # Gửi frames và prompt đến Gemini
                contents = [prompt]
                
                # Thêm từng frame dưới dạng hình ảnh
                for frame in frame_base64:
                    contents.append(
                        types.Part(inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(frame)
                        ))
                    )
                
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["Text"]
                    ),
                )
                
                analysis_result = ""
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if part.text:
                            analysis_result = part.text
                            break
                    if analysis_result:
                        break
                
                # Store analysis result
                video_analysis_results[filename] = analysis_result
                
                return jsonify({
                    'success': True,
                    'analysis': analysis_result
                })
            
        except Exception as api_error:
            print(f"[DEBUG] Error calling API: {str(api_error)}")
            
            # Tạo phân tích lỗi mẫu nếu API lỗi
            fallback_analysis = f"""PRODUCT NAME: Website Analysis

OVERALL ANALYSIS:
Không thể phân tích video do gặp lỗi từ API. Đây có thể là do kích thước video quá lớn hoặc định dạng không được hỗ trợ.

STRENGTHS:
- Không có thông tin

WEAKNESSES:
- Không có thông tin

IMPROVEMENT SUGGESTIONS:
- Thử ghi lại video với thời gian ngắn hơn
- Thử với URL khác có ít nội dung hơn

Lỗi chi tiết: {str(api_error)}
"""
            
            # Lưu kết quả phân tích lỗi
            video_analysis_results[filename] = fallback_analysis
            
            return jsonify({
                'success': True,
                'analysis': fallback_analysis,
                'error_info': str(api_error)
            })
    
    except Exception as e:
        print(f"[DEBUG] Error analyzing video: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# Load analytics from file when starting up
def load_analytics():
    global store_analytics, next_store_analytics_id
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                store_analytics = data.get('analytics', {})
                next_store_analytics_id = data.get('next_id', 1)
                print(f"[STARTUP] Loaded {len(store_analytics)} analytics records")
    except Exception as e:
        print(f"[STARTUP] Error loading analytics: {str(e)}")
        store_analytics = {}
        next_store_analytics_id = 1

# Save analytics to file
def save_analytics():
    try:
        with open(ANALYTICS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'analytics': store_analytics,
                'next_id': next_store_analytics_id
            }, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Saved {len(store_analytics)} analytics records")
    except Exception as e:
        print(f"[SAVE] Error saving analytics: {str(e)}")

# Load analytics when starting up
load_analytics()

@app.route('/store_analytics', methods=['GET', 'POST'])
def store_analytics_endpoint():
    global store_analytics
    
    if request.method == 'GET':
        # Chuyển đổi dictionary thành list và sắp xếp theo timestamp giảm dần
        analytics_list = list(store_analytics.values())
        analytics_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(analytics_list)
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['id', 'url', 'product_name', 'analysis', 'video_path']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'success': False,
                        'message': f'Missing required field: {field}'
                    }), 400
            
            # Add timestamp if not provided
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().timestamp()
            
            # Add creation_time if not provided
            if 'creation_time' not in data:
                data['creation_time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            
            # Store the analysis
            store_analytics[data['id']] = data
            
            # Save to file
            save_analytics()
            
            return jsonify({
                'success': True,
                'message': 'Analysis stored successfully'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 500

@app.route('/store_analysis/<analysis_id>', methods=['GET'])
def get_store_analysis(analysis_id):
    try:
        if analysis_id in store_analytics:
            return jsonify({
                'success': True,
                'analysis': store_analytics[analysis_id]['analysis']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Analysis not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def extract_scores_from_text(analysis_text):
    """
    Trích xuất điểm số và thông tin từ kết quả phân tích
    """
    criteria_scores = []
    total_score = 0
    grade = "Unknown"
    
    # Tìm tổng điểm
    for line in analysis_text.split('\n'):
        if "Tổng điểm:" in line:
            try:
                total_score = int(line.split(":")[1].split("/")[0].strip())
                break
            except:
                continue
    
    # Tìm xếp loại
    for line in analysis_text.split('\n'):
        if "Xếp loại:" in line:
            grade = line.split(":")[1].strip()
            break
    
    # Trích xuất điểm từng tiêu chí
    current_criterion = None
    for line in analysis_text.split('\n'):
        if "Tiêu chí đánh giá" in line:
            parts = line.split("\t")
            if len(parts) >= 4:
                criterion_name = parts[1].strip()
                explanation = parts[2].strip()
                try:
                    score = int(parts[3].strip().replace("[", "").replace("]", ""))
                    criteria_scores.append({
                        "name": criterion_name,
                        "score": score,
                        "explanation": explanation
                    })
                except:
                    continue
    
    return criteria_scores, total_score, grade

@app.route('/analyze_store', methods=['POST'])
def analyze_store():
    global next_store_analytics_id
    try:
        print("\n[ANALYZE-STORE] Starting store analysis process...")
        data = request.get_json()
        url = data.get('url')
        print(f"[ANALYZE-STORE] URL to analyze: {url}")
        
        if not url:
            print("[ANALYZE-STORE] ERROR: URL is required")
            return jsonify({'error': 'URL is required'}), 400
            
        # Bước 1: Ghi lại trang web
        print(f"[ANALYZE-STORE] Step 1: Recording webpage...")
        recording_start_time = time.time()
        output_path = record_webpage(url)
        recording_end_time = time.time()
        recording_duration = recording_end_time - recording_start_time
        print(f"[ANALYZE-STORE] Recording completed in {recording_duration:.2f} seconds")
        
        if not output_path:
            print("[ANALYZE-STORE] ERROR: Failed to record webpage")
            return jsonify({
                'success': False,
                'message': 'Failed to record webpage'
            }), 500
            
        # Bước 2: Kiểm tra và xử lý video
        print(f"[ANALYZE-STORE] Step 2: Checking video file...")
        if not os.path.exists(output_path):
            print(f"[ANALYZE-STORE] ERROR: Video file not found at: {output_path}")
            return jsonify({
                'success': False,
                'message': f'Video file not found at: {output_path}'
            }), 404
            
        # Kiểm tra kích thước file
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ANALYZE-STORE] Video file size: {file_size_mb:.2f} MB")
        
        # Bước 3: Phân tích video
        print(f"[ANALYZE-STORE] Step 3: Analyzing video...")
        analysis_start_time = time.time()
        
        # Tạo prompt cho phân tích
        prompt = """ Bạn là chuyên gia về tối ưu trang sản phẩm e-commerce. Hãy phân tích kỹ cấu trúc và nội dung của trang sản phẩm hiện tại.

            **Yêu cầu chi tiết (trả lời hoàn toàn bằng tiếng Việt):**

            1. **Nhận diện các mục trên trang sản phẩm**:
                - Liệt kê và đặt tên rõ ràng, logic cho từng phần nội dung có trên trang sản phẩm hiện tại (VD: Banner chính, Mô tả sản phẩm, Đánh giá khách hàng, v.v.)

            2. **Đánh giá từng mục**:
                Với mỗi mục vừa nhận diện, đưa ra:
                - Điểm đánh giá từ 1-10 dựa trên các tiêu chí: rõ ràng, thuyết phục, tính liên quan, hiệu quả SEO, khả năng chuyển đổi.
                - Phân tích ngắn gọn về điểm mạnh và điểm yếu.
                - Đề xuất cải thiện (nếu có).

            3. **Benchmarking (So sánh chuẩn)**:
                - So sánh cấu trúc trang sản phẩm hiện tại với các trang sản phẩm hiệu quả cao (VD: Amazon, Shopify, các thương hiệu DTC nổi bật).
                - Tóm tắt rõ ràng các phần thiếu hoặc thừa trên trang hiện tại.

            4. **Đề xuất cấu trúc trang sản phẩm tối ưu**:
                - Đề xuất cấu trúc lý tưởng (tên từng mục, thứ tự sắp xếp và mục đích) nhằm tối ưu hóa SEO và tăng tỷ lệ chuyển đổi.

            5. **Viết lại nội dung sản phẩm**:
                - Dựa trên đánh giá và cấu trúc đề xuất, viết lại nội dung trang sản phẩm.
                - Đảm bảo nội dung mới tối ưu SEO, thúc đẩy chuyển đổi, và giữ nguyên phong cách thương hiệu.

            6. **Gợi ý làm video giới thiệu sản phẩm**:
                - Đưa ra một kịch bản video ngắn khoảng 1 phút, giới thiệu sản phẩm theo phong cách hấp dẫn, hiệu quả.
                - Do đây là sản phẩm dropshipping, hiện tại chưa có sản phẩm trong tay, hãy gợi ý cụ thể cách làm video mà không cần sản phẩm vật lý (VD: sử dụng hình ảnh, video mẫu, đồ họa, stock footage).

            ---

            **Yêu cầu đặc biệt về định dạng đầu ra**:
            Trả lời toàn bộ nội dung ở trên bằng tiếng Việt và theo định dạng JSON như sau:

            ```json
            {
                "nhan_dien_cac_muc": [
                    {
                        "ten_muc": "Tên mục cụ thể",
                        "diem_danh_gia": 1-10,
                        "phan_tich": "Điểm mạnh và điểm yếu",
                        "goi_y_cai_thien": "Cách cải thiện mục này"
                    }
                ],
                "benchmarking": {
                    "phan_thieu": ["Danh sách các mục thiếu"],
                    "phan_thua": ["Danh sách các mục thừa"],
                    "so_sanh_chung": "Tóm tắt so sánh ngắn gọn"
                },
                "cau_truc_trang_toi_uu": [
                    {
                        "ten_muc": "Tên mục đề xuất",
                        "thu_tu": "Vị trí mục trong trang (VD: 1, 2, 3...)",
                        "muc_dich": "Mục đích cụ thể của mục này"
                    }
                ],
                "noi_dung_viet_lai": "Toàn bộ nội dung sản phẩm đã được viết lại, tối ưu SEO, chuyển đổi và giữ nguyên giọng thương hiệu",
                "goi_y_video_1_phut": {
                    "kich_ban": "Kịch bản cụ thể video trong 1 phút",
                    "cach_thuc_thuc_hien": "Cách làm video mà không có sản phẩm vật lý (stock video, hình ảnh minh họa, motion graphics,...)"
                }
            }

        """

        
        # Gọi API analyze_video để phân tích
        print("[ANALYZE-STORE] Calling analyze_video function...")
        analysis_result = analyze_video(output_path, prompt)
        
        if analysis_result.get('error'):
            print(f"[ANALYZE-STORE] ERROR: {analysis_result['analysis']}")
            return jsonify({
                'success': False,
                'message': analysis_result['analysis']
            }), 500
            
        analysis_end_time = time.time()
        analysis_duration = analysis_end_time - analysis_start_time
        print(f"[ANALYZE-STORE] Analysis completed in {analysis_duration:.2f} seconds")
        
        # Bước 4: Trích xuất tên sản phẩm và lưu kết quả
        print("[ANALYZE-STORE] Step 4: Extracting product name and saving results...")
        product_name = "Website Analysis"
        for line in analysis_result['analysis'].split('\n'):
            if line.startswith("PRODUCT NAME:"):
                product_name = line.replace("PRODUCT NAME:", "").strip()
                break
        print(f"[ANALYZE-STORE] Extracted product name: {product_name}")
                
        # Tạo ID duy nhất cho phân tích này
        analysis_id = str(next_store_analytics_id)
        next_store_analytics_id += 1
        print(f"[ANALYZE-STORE] Generated analysis ID: {analysis_id}")
        
        # Lưu phân tích với metadata
        store_analytics[analysis_id] = {
            'id': analysis_id,
            'url': url,
            'product_name': product_name,
            'analysis': analysis_result['analysis'],
            'video_path': f'/output/{os.path.basename(output_path)}',
            'creation_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'timestamp': datetime.now().timestamp()
        }
        print("[ANALYZE-STORE] Analysis saved successfully")
        
        # Save to file after adding new analysis
        save_analytics()
        
        # Trích xuất điểm số và thông tin cho dashboard
        print("[ANALYZE-STORE] Step 5: Extracting scores for dashboard...")
        criteria_scores, total_score, grade = extract_scores_from_text(analysis_result['analysis'])
        
        print("[ANALYZE-STORE] Process completed successfully!")
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'product_name': product_name,
            'url': url,
            'creation_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'score': total_score,
            'grade': grade
        })
            
    except Exception as e:
        print(f"[ANALYZE-STORE] ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/view_report/<analysis_id>')
def view_report(analysis_id):
    """
    Hiển thị dashboard cho một phân tích cụ thể
    """
    try:
        if analysis_id not in store_analytics:
            return "Analysis not found", 404
            
        analysis = store_analytics[analysis_id]
        criteria_scores, total_score, grade = extract_scores_from_text(analysis['analysis'])
        
        return render_template("dashboard.html",
            product_name=analysis['product_name'],
            url=analysis['url'],
            score=total_score,
            grade=grade,
            criteria=criteria_scores,
            analysis_summary=analysis['analysis']
        )
    except Exception as e:
        print(f"[VIEW-REPORT] ERROR: {str(e)}")
        return str(e), 500

def analyze_video(video_path, custom_prompt=""):
    """
    Phân tích video sử dụng Google Gemini API.
    Cho phép sử dụng prompt tùy chỉnh nếu được cung cấp.
    """
    print("\n[ANALYZE-VIDEO] Starting video analysis process...")
    print(f"[ANALYZE-VIDEO] Video path: {video_path}")
    
    try:
        # Đảm bảo đường dẫn video hợp lệ
        absolute_video_path = os.path.abspath(video_path)
        print(f"[ANALYZE-VIDEO] Absolute video path: {absolute_video_path}")
        
        if not os.path.exists(absolute_video_path):
            print(f"[ANALYZE-VIDEO] ERROR: Video file not found at {absolute_video_path}")
            return {
                "product_name": "Unknown",
                "analysis": f"Error: Video file not found at {absolute_video_path}",
                "error": True
            }
        
        # Đọc video file
        print("[ANALYZE-VIDEO] Reading video file...")
        with open(absolute_video_path, 'rb') as video_file:
            video_data = video_file.read()
        print(f"[ANALYZE-VIDEO] Video file size: {len(video_data) / (1024*1024):.2f} MB")
        
        # Xác định MIME type
        mime_type = "video/mp4"
        if absolute_video_path.lower().endswith('.avi'):
            mime_type = "video/avi"
        print(f"[ANALYZE-VIDEO] MIME type: {mime_type}")
        
        # Tạo prompt mặc định nếu không có prompt tùy chỉnh
        if not custom_prompt:
            custom_prompt = """Hãy phân tích trang web bán sản phẩm này và cung cấp thông tin sau:
            1. Sản phẩm chính được bán là gì?
            2. Đối tượng khách hàng mục tiêu mà trang web hướng đến là ai?
            3. Trải nghiệm người dùng trên trang web này như thế nào? (Điều hướng, bố cục, tốc độ tải ước tính, tính thân thiện trên thiết bị di động)
            4. Thông tin về sản phẩm (mô tả, hình ảnh, giá cả) được trình bày như thế nào?
            5. Chính sách bán hàng và vận chuyển có rõ ràng và dễ hiểu không?
            6. Mức độ tin cậy của trang web (dựa trên các yếu tố có thể thấy trên trang).
            7. Đề xuất cải thiện để tăng doanh số và trải nghiệm người dùng (tối thiểu 3 đề xuất cụ thể)"""
            print("[ANALYZE-VIDEO] Using default prompt")
        else:
            print("[ANALYZE-VIDEO] Using custom prompt")
        
        # In ra prompt sẽ được sử dụng
        print("\n[ANALYZE-VIDEO] Prompt to be sent to Gemini:")
        print("-" * 50)
        print(custom_prompt)
        print("-" * 50)
        
        # Tạo contents array với prompt và video
        print("\n[ANALYZE-VIDEO] Preparing content for Gemini API...")
        contents = [
            custom_prompt,
            types.Part(inline_data=types.Blob(
                mime_type=mime_type,
                data=video_data
            ))
        ]
        
        # In ra thông tin về contents
        print(f"[ANALYZE-VIDEO] Content structure:")
        print(f"- Number of parts: {len(contents)}")
        print(f"- Part 1: Text prompt (length: {len(custom_prompt)} characters)")
        print(f"- Part 2: Video data (size: {len(video_data) / (1024*1024):.2f} MB)")
        
        # Gửi video đến Gemini
        print("\n[ANALYZE-VIDEO] Sending request to Gemini API...")
        print(f"[ANALYZE-VIDEO] Model: gemini-2.0-flash-exp")
        print(f"[ANALYZE-VIDEO] Response modalities: ['Text']")
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["Text"]
            ),
        )
        print("[ANALYZE-VIDEO] Received response from Gemini API")
        
        # Lấy kết quả phân tích
        print("\n[ANALYZE-VIDEO] Processing response...")
        analysis_result = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.text:
                    analysis_result = part.text
                    break
            if analysis_result:
                break
        
        print(f"[ANALYZE-VIDEO] Analysis result length: {len(analysis_result)} characters")
        print("\n[ANALYZE-VIDEO] Analysis result preview (first 200 chars):")
        print("-" * 50)
        print(analysis_result[:200] + "...")
        print("-" * 50)
        print("[ANALYZE-VIDEO] Analysis completed successfully!")
        
        return {
            "product_name": "Video Analysis",
            "analysis": analysis_result,
            "error": False
        }
        
    except Exception as e:
        print(f"[ANALYZE-VIDEO] ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "product_name": "Unknown",
            "analysis": f"Error analyzing video: {str(e)}",
            "error": True
        }

@app.route('/analyze_uploaded_video', methods=['POST'])
def analyze_uploaded_video():   
    """
    Phân tích video đã tải lên bằng Gemini.
    """
    try:
        video_path = request.json.get('video_path')
        if not video_path or video_path not in UPLOADED_VIDEOS:
            return jsonify({'success': False, 'message': 'Invalid video path'}), 400
        
        # Nếu đã có kết quả phân tích trước đó, trả về luôn
        if UPLOADED_VIDEOS[video_path]['has_analysis'] and UPLOADED_VIDEOS[video_path]['analysis']:
            return jsonify({
                'success': True,
                'analysis': UPLOADED_VIDEOS[video_path]['analysis']
            })
        
        # Lấy prompt tùy chỉnh nếu có
        custom_prompt = UPLOADED_VIDEOS[video_path].get('analysis_prompt', '')
        
        # Phân tích video bằng hàm analyze_video
        print(f"Analyzing uploaded video: {video_path}")
        analysis_result = analyze_video(video_path, custom_prompt)
        
        if analysis_result.get('error'):
            return jsonify({
                'success': False,
                'message': analysis_result['analysis']
            }), 500
        
        # Lưu kết quả phân tích
        UPLOADED_VIDEOS[video_path]['has_analysis'] = True
        UPLOADED_VIDEOS[video_path]['analysis'] = analysis_result['analysis']
        
        return jsonify({
            'success': True,
            'analysis': analysis_result['analysis']
        })
    
    except Exception as e:
        print(f"Error analyzing uploaded video: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'videoFile' not in request.files:
            return jsonify({'success': False, 'message': 'No video file provided'}), 400
        
        video_file = request.files['videoFile']
        if video_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400
        
        if video_file and allowed_file(video_file.filename):
            filename = secure_filename(video_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            video_file.save(file_path)
            
            # Lưu thông tin video
            UPLOADED_VIDEOS[file_path] = {
                'name': filename,
                'path': file_path,
                'has_analysis': False,
                'analysis': None,
                'analysis_prompt': request.form.get('analysisPrompt', ''),
                'upload_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            
            return jsonify({
                'success': True,
                'message': 'Video uploaded successfully',
                'video_path': file_path
            })
        
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400
    
    except Exception as e:
        print(f"Error uploading video: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/uploaded_videos', methods=['GET'])
def get_uploaded_videos():
    try:
        videos = []
        for video_path, video_info in UPLOADED_VIDEOS.items():
            if os.path.exists(video_path):
                videos.append({
                    'name': video_info['name'],
                    'path': video_info['path'],
                    'has_analysis': video_info['has_analysis'],
                    'upload_time': video_info['upload_time']
                })
        return jsonify(videos)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/optimize_image', methods=['POST'])
def optimize_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image file provided'})
        
        image_file = request.files['image']
        prompt = request.form.get('prompt', '')
        
        if image_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
        
        # Read the image file
        image_data = image_file.read()
        
        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare the prompt for Gemini
        full_prompt = f"{prompt}\n\nPlease process this image and return the result as a base64-encoded PNG image."
        
        # Call Gemini API with correct syntax
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                full_prompt,
                types.Part(inline_data=types.Blob(
                    mime_type="image/png",
                    data=image_data
                ))
            ],
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )
        
        # Extract the base64 image from the response
        result_image = None
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.mime_type == "image/png":
                    result_image = base64.b64encode(part.inline_data.data).decode('utf-8')
                    break
            if result_image:
                break
        
        if not result_image:
            return jsonify({'success': False, 'message': 'No image response from Gemini API'})
        
        # Lưu kết quả vào optimize_image_data
        result_id = str(optimize_image_data['next_id'])
        optimize_image_data['next_id'] += 1
        
        optimize_image_data['results'][result_id] = {
            'id': result_id,
            'original_image': image_base64,
            'result_image': result_image,
            'prompt': prompt,
            'timestamp': datetime.now().timestamp(),
            'creation_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        
        # Lưu vào file
        save_optimize_image_data()
        
        return jsonify({
            'success': True,
            'image': result_image,
            'id': result_id
        })
            
    except Exception as e:
        print(f"Error in optimize_image: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/optimize_background_with_ai', methods=['POST'])
def optimize_background_with_ai():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image file provided'})
        
        image_file = request.files['image']
        prompt = request.form.get('prompt', '')
        
        if image_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
        
        # Read the image file
        image_data = image_file.read()
        
        # Call Gemini API with the generated prompt
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                prompt,
                types.Part(inline_data=types.Blob(
                    mime_type="image/png",
                    data=image_data
                ))
            ],
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )
        
        # Extract the base64 image from the response
        result_image = None
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.mime_type == "image/png":
                    result_image = base64.b64encode(part.inline_data.data).decode('utf-8')
                    break
            if result_image:
                break
        
        if not result_image:
            return jsonify({'success': False, 'message': 'No image response from Gemini API'})
        
        # Lưu kết quả vào optimize_image_data
        result_id = str(optimize_image_data['next_id'])
        optimize_image_data['next_id'] += 1
        
        optimize_image_data['results'][result_id] = {
            'id': result_id,
            'original_image': base64.b64encode(image_data).decode('utf-8'),
            'result_image': result_image,
            'prompt': prompt,
            'timestamp': datetime.now().timestamp(),
            'creation_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        
        # Lưu vào file
        save_optimize_image_data()
        
        return jsonify({
            'success': True,
            'image': result_image,
            'id': result_id
        })
            
    except Exception as e:
        print(f"Error in optimize_background_with_ai: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/optimize_image_results', methods=['GET'])
def get_optimize_image_results():
    # Convert results dictionary to list and sort by timestamp
    results_list = []
    for result_id, result in optimize_image_data['results'].items():
        result['id'] = result_id
        results_list.append(result)
    
    # Sort by timestamp in descending order (newest first)
    results_list.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(results_list)

@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        # Get uploaded files
        images = request.files.getlist('images')
        image_urls = request.form.getlist('image_urls')
        
        if not images and not image_urls:
            return jsonify({'success': False, 'message': 'No images provided'})
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(app.root_path, 'static', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'video_{timestamp}.mp4')
        
        # Process images
        image_files = []
        
        # Process uploaded files
        for image in images:
            if image.filename:
                image_files.append(image)
        
        # Process URLs
        for url in image_urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Create a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_file.write(response.content)
                    temp_file.close()
                    image_files.append(open(temp_file.name, 'rb'))
            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")
        
        if not image_files:
            return jsonify({'success': False, 'message': 'No valid images to process'})
        
        # Create video from images
        first_image = Image.open(image_files[0])
        width, height = first_image.size
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 1.0, (width, height))
        
        # Add each image to the video
        for image_file in image_files:
            # Convert PIL Image to OpenCV format
            img = Image.open(image_file)
            img = img.convert('RGB')
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(img_cv)
            
            # Add a black frame for transition
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(black_frame)
        
        # Release video writer
        out.release()
        
        # Convert to MP4 using ffmpeg
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        os.rename(output_path, temp_output)
        
        subprocess.run([
            'ffmpeg', '-i', temp_output,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            output_path
        ])
        
        # Remove temporary file
        os.remove(temp_output)
        
        # Return the video path
        return jsonify({
            'success': True,
            'video_path': f'/static/output/video_{timestamp}.mp4'
        })
        
    except Exception as e:
        print(f"Error in generate_video: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/generate_ai_prompt', methods=['POST'])
def generate_ai_prompt():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'No image file provided'})
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})
        
        # Read the image file
        image_data = image_file.read()
        print("\n[GEMINI-API] Image size:", len(image_data), "bytes")
        
        # Prepare the prompt for Gemini
        prompt = """Bạn hãy xem bức ảnh này và xác định sản phẩm trong ảnh là gì, đồng thời phân tích key feature nổi bật nhất của sản phẩm đó (bao gồm công năng, chất liệu, đối tượng sử dụng, phong cách thiết kế). Sau đó, hãy tạo một prompt để tối ưu lại hình ảnh sản phẩm, dựa trên key feature vừa phân tích.

⚠️ Lưu ý quan trọng:
Chỉ chọn bối cảnh tối ưu phù hợp với mục đích sử dụng thực tế và cảm xúc khách hàng mục tiêu.
Không đặt sản phẩm vào bối cảnh sai công năng (ví dụ: móc khóa thú bông dùng để treo túi, ba lô hoặc sưu tập — không phù hợp nếu đặt trên bàn làm việc văn phòng).

⚠️ Cảnh báo thêm:
Với các sản phẩm trang sức hoặc phụ kiện đeo trên người (ví dụ: nhẫn, vòng cổ, vòng tay, bông tai), hình ảnh cần có người mẫu thật với kết cấu da rõ ràng, ánh sáng thật, bóng đổ tự nhiên, tóc thật hoặc được xử lý tinh tế.
Tránh mọi hình ảnh “mịn bất thường”, không có lỗ chân lông, thiếu chi tiết da, thiếu shadow, hoặc bị “AI hoá” gây cảm giác không chân thực. Ví dụ đúng 1 – Móc khóa thú bông:
"Hãy tối ưu lại background của sản phẩm này thành một chiếc balo học sinh hoặc túi xách nữ trong bối cảnh đời sống thường ngày như trường học, quán cà phê, hoặc chuyến du lịch. Treo sản phẩm ở vị trí dễ nhìn, thể hiện rõ kích thước thật, giữ nguyên màu sắc và chất liệu lông bông mềm mại. Ánh sáng chân thực, không dùng hiệu ứng hoạt hình hay nhân tạo.
Với các sản phẩm có kích thước cụ thể như túi xách, balo, tranh, đèn, đồ nội thất, phụ kiện đeo, hình ảnh phải thể hiện đúng tỷ lệ kích thước thực tế khi đặt cạnh người hoặc vật thể xung quanh (tay, người, bàn ghế...).
Không được phóng to hoặc thu nhỏ gây sai lệch cảm nhận về kích thước thật, làm giảm độ tin cậy sản phẩm.    "

Ví dụ đúng 2 – Nhẫn kim cương:
"Hãy đặt chiếc nhẫn kim cương này lên ngón tay của một người thật trong bối cảnh buổi hẹn tối lãng mạn tại nhà hàng sang trọng. Tay người phải chân thực, có kết cấu da tự nhiên, ánh sáng ấm nhẹ từ nến hoặc đèn tạo phản chiếu lấp lánh lên nhẫn. Nhẫn cần được chụp rõ nét, nổi bật trên tay, thể hiện được độ chi tiết của đá và ánh kim loại. Tổng thể hình ảnh phải mang phong cách chụp ảnh thực tế (photorealistic), không sử dụng hiệu ứng hoạt hình hoặc phong cách AI."

Ví dụ đúng 3 – Vòng cổ:
"Hãy đặt chiếc vòng cổ này lên cổ của một người phụ nữ trong bối cảnh dự tiệc buổi tối hoặc sự kiện trang trọng. Ánh sáng dịu nhẹ làm nổi bật chất liệu kim loại và đá quý. Cổ và phần da xung quanh phải chân thực, có kết cấu rõ ràng. Chiếc vòng cần được chụp cận để thể hiện sự tinh xảo của thiết kế và độ bóng sáng."

Ví dụ đúng 4 – Bông tai (cơ bản):
"Hãy đeo đôi bông tai này lên tai của một người thật, trong bối cảnh ánh sáng tự nhiên hoặc khung cảnh tiệc cưới, fashion show, hoặc street style hiện đại. Khuôn mặt hoặc góc nghiêng cần thể hiện rõ phần tai, tóc thật và làn da có kết cấu chân thực. Bông tai cần được làm nổi bật về chất liệu, hình dáng và độ long lanh. Ảnh mang phong cách chụp thời trang cao cấp, không dùng hiệu ứng ảo."

Ví dụ đúng 9 – Bông tai (nâng cao – chống AI look):
"Hãy đeo đôi bông tai này lên tai của một người thật trong bối cảnh ánh sáng dịu nhẹ của một buổi tiệc tối hoặc sự kiện đặc biệt. Tóc nên được búi cao hoặc vén gọn để lộ toàn bộ tai. Khuôn mặt hoặc góc nghiêng cần thể hiện rõ kết cấu da thật như lỗ chân lông, ánh sáng phản chiếu nhẹ và bóng đổ tự nhiên. Bông tai phải gắn rõ ràng, nổi bật trên tai. Hình ảnh cần mang phong cách chụp chân dung thật (photorealistic), không làm mịn da quá mức, không sử dụng hiệu ứng hoạt hình hoặc AI."

Ví dụ đúng 5 – Tranh treo tường:
"Hãy treo bức tranh này lên tường của một không gian phòng khách hiện đại hoặc studio nghệ thuật. Tường nên có chất liệu gạch, bê tông hoặc gỗ, ánh sáng nhẹ từ cửa sổ hoặc đèn tường làm nổi bật chi tiết và màu sắc tranh. Tranh phải được hiển thị đúng tỷ lệ với không gian và nội thất xung quanh, tạo cảm giác hài hòa và sang trọng."

Ví dụ đúng 6 – Vòng tay phong cách:
"Hãy đeo chiếc vòng tay này lên cổ tay của một người thật trong khung cảnh đời sống hằng ngày như quán cà phê, buổi hẹn, hoặc khi đang dùng điện thoại. Tay và làn da phải chân thực, có kết cấu rõ ràng. Vòng tay cần được làm nổi bật ở vị trí trung tâm, ánh sáng tự nhiên phản chiếu nhẹ nhàng để thấy rõ chất liệu và chi tiết thiết kế."

Ví dụ đúng 7 – Đèn decor nghệ thuật:
"Hãy đặt chiếc đèn decor này trong một không gian nội thất như phòng khách, phòng ngủ hoặc studio cá nhân. Bối cảnh cần có ánh sáng dịu để đèn phát sáng rõ ràng, làm nổi bật hình dáng và màu sắc. Giữ đúng tỉ lệ giữa đèn và đồ nội thất xung quanh để thể hiện công năng sử dụng và tạo cảm giác ấm áp, phong cách sống hiện đại."

Ví dụ đúng 8 – Đồ sưu tầm nhỏ (figure, mô hình, đồ chơi nghệ thuật):
"Hãy trưng bày món đồ sưu tầm này trên kệ gỗ, tủ kính hoặc bàn trưng bày có đèn chiếu nhẹ, trong một không gian cá nhân như góc làm việc hoặc phòng trưng bày nghệ thuật. Giữ nguyên kích thước thực tế, thể hiện rõ chất liệu và chi tiết sản phẩm. Bối cảnh cần gợi cảm hứng 'sưu tầm' và đam mê cá nhân, ảnh mang phong cách tự nhiên – không hoạt hình hóa."

Ví dụ đúng 9 – Túi xách thời trang:
"Hãy đeo chiếc túi xách này trên vai hoặc cầm tay bởi một người thật đang di chuyển trong bối cảnh đời thường như trên phố, trước quán cà phê, hoặc ở trường học. Túi cần được hiển thị đúng tỷ lệ thực tế so với cơ thể người mẫu, không bị phóng to hoặc thu nhỏ bất thường. Ánh sáng tự nhiên chiếu vào chất liệu vải, thể hiện rõ kết cấu, màu sắc và độ đứng dáng của túi. Hình ảnh cần chân thực, không dùng hiệu ứng hoạt hình, filter làm ảo hoặc phong cách AI."

Ví dụ sai cần tránh:
"Hãy đặt sản phẩm lên bàn làm việc với sách vở và tách cà phê" (sai công năng, không hợp lý với sản phẩm dùng để treo hoặc mang theo).

Yêu cầu cuối cùng: phần kết quả trả về cho tôi chỉ cần đúng một dòng prompt như ví dụ đúng trên, không cần phân tích hay diễn giải thêm."""
        
        print("\n[GEMINI-API] Sending request to Gemini:")
        print("Model: gemini-2.0-flash-exp")
        print("Prompt length:", len(prompt), "characters")
        print("Image size:", len(image_data), "bytes")
        
        # Call Gemini API to generate prompt
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                prompt,
                types.Part(inline_data=types.Blob(
                    mime_type="image/png",
                    data=image_data
                ))
            ],
            config=types.GenerateContentConfig(
                response_modalities=["Text"]
            )
        )
        
        print("\n[GEMINI-API] Received response from Gemini:")
        print("Response type:", type(response))
        print("Has candidates:", hasattr(response, 'candidates'))
        if hasattr(response, 'candidates'):
            print("Number of candidates:", len(response.candidates))
        
        # Extract the prompt from the response
        generated_prompt = ""
        try:
            if response and hasattr(response, 'candidates'):
                for candidate in response.candidates:
                    print("\nProcessing candidate:")
                    print("Has content:", hasattr(candidate, 'content'))
                    if hasattr(candidate, 'content') and candidate.content:
                        print("Has parts:", hasattr(candidate.content, 'parts'))
                        if hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                print("Has text:", hasattr(part, 'text'))
                                if hasattr(part, 'text') and part.text:
                                    generated_prompt = part.text.strip()
                                    print("Found text:", generated_prompt[:100] + "..." if len(generated_prompt) > 100 else generated_prompt)
                                    break
                            if generated_prompt:
                                break
                    if generated_prompt:
                        break
        except Exception as e:
            print(f"\n[GEMINI-API] Error extracting prompt from response: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Error extracting prompt from API response'
            })
        
        if not generated_prompt:
            print("\n[GEMINI-API] No prompt generated from the API")
            return jsonify({
                'success': False,
                'message': 'No prompt generated from the API'
            })
        
        # Extract the actual prompt from the response
        try:
            # Tìm vị trí của prompt trong text
            prompt_start = generated_prompt.lower().find("prompt:")
            if prompt_start != -1:
                # Lấy phần text sau "prompt:"
                generated_prompt = generated_prompt[prompt_start + 7:].strip()
                # Loại bỏ dấu ngoặc kép nếu có
                if generated_prompt.startswith('"') and generated_prompt.endswith('"'):
                    generated_prompt = generated_prompt[1:-1]
                print("\n[GEMINI-API] Processed prompt:", generated_prompt)
        except Exception as e:
            print(f"\n[GEMINI-API] Error processing prompt text: {str(e)}")
            # If we can't process the prompt, return it as is
            pass
        
        return jsonify({
            'success': True,
            'prompt': generated_prompt
        })
            
    except Exception as e:
        print(f"\n[GEMINI-API] Error in generate_ai_prompt: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error generating prompt: {str(e)}'
        })

# Load optimize image data from file when starting up
def load_optimize_image_data():
    global optimize_image_data
    try:
        if os.path.exists(OPTIMIZE_IMAGE_FILE):
            with open(OPTIMIZE_IMAGE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                optimize_image_data = data
                print(f"[STARTUP] Loaded {len(optimize_image_data['results'])} optimize image records")
    except Exception as e:
        print(f"[STARTUP] Error loading optimize image data: {str(e)}")
        optimize_image_data = {'next_id': 1, 'results': {}}

# Save optimize image data to file
def save_optimize_image_data():
    try:
        with open(OPTIMIZE_IMAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(optimize_image_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Saved {len(optimize_image_data['results'])} optimize image records")
    except Exception as e:
        print(f"[SAVE] Error saving optimize image data: {str(e)}")

if __name__ == "__main__":
    # Load optimize image data when starting up
    load_optimize_image_data()
    app.run(debug=True, port=5002)

