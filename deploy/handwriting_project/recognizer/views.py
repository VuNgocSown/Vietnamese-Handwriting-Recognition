from django.shortcuts import render
from django.http import JsonResponse
import torch
import os
import tempfile
from .decode import decode_prediction, num_classes
from .architecture import preprocess, segment_lines, CRNN
import cv2
import numpy as np
import base64
from concurrent.futures import ThreadPoolExecutor
import traceback

# Đường dẫn đến file trọng số
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights.pth')

def index(request):
    return render(request, 'recognizer/index.html')

# Tải model 1 lần ở module (CPU)
device = torch.device('cpu')
model = CRNN(num_classes)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

def recognize_handwriting(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are accepted'}, status=405)

    # Remove broad try/except to allow Django to display full traceback for debugging
    # --- Preview segmentation for upload (only boxes, no text) ---
    if request.POST.get('preview') == '1':
        # Handle file upload
        if 'image_file' in request.FILES:
            image_file = request.FILES['image_file']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                for chunk in image_file.chunks(): temp_file.write(chunk)
                temp_path = temp_file.name
        else:
            return JsonResponse({'error': 'Invalid preview request'}, status=400)
        # Perform segmentation
        try:
            _, boxes_info, img_shape = segment_lines(temp_path)
            os.unlink(temp_path)
            return JsonResponse({'segments': boxes_info,
                                 'imageWidth': img_shape[1],
                                 'imageHeight': img_shape[0]})
        except Exception as e:
            os.unlink(temp_path)
            return JsonResponse({'error': f'Segmentation failed: {str(e)}'}, status=500)
    # -----------------------------------------------------------
    if 'image_file' not in request.FILES and 'image_data' not in request.POST:
        return JsonResponse({'error': 'No image data provided'}, status=400)
    
    # Get iteration parameter (default to 1 if not provided)
    iteration = int(request.POST.get('iteration', 1))
    
    # Handle regular file upload or base64 image from previous iteration
    if 'image_file' in request.FILES:
        image_file = request.FILES['image_file']
        
        # Lưu file tạm thời để xử lý
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
    else:
        # Process base64 image data from previous iteration
        image_data = request.POST.get('image_data')
        if not image_data:
            return JsonResponse({'error': 'Invalid image data'}, status=400)
        
        # Remove prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Decode base64 to binary
        image_binary = base64.b64decode(image_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_binary)
            temp_path = temp_file.name
    
    # Phân đoạn ảnh thành các box và lấy thông tin vị trí
    try:
        line_images, boxes_info, img_shape = segment_lines(temp_path)
        
        if not line_images:
            os.unlink(temp_path)
            return JsonResponse({'error': 'No text lines detected in image'}, status=400)
    except Exception as e:
        os.unlink(temp_path)
        return JsonResponse({'error': f'Line segmentation failed: {str(e)}'}, status=400)
    
    # Đọc ảnh gốc để lấy kích thước, kiểm tra lỗi đọc ảnh
    try:
        original_img = cv2.imread(temp_path)
        if original_img is None:
            raise ValueError("cv2.imread returned None")
        img_height, img_width = original_img.shape[:2]
    except Exception as e:
        os.unlink(temp_path)
        return JsonResponse({'error': f'Failed to read image: {str(e)}'}, status=500)
    
    # Xử lý từng box (đa luồng, tối ưu cho CPU)
    def process_line(args):
        i, line_img = args
        torch.set_num_threads(1)
        preprocessed_image = preprocess(line_img)
        input_tensor = torch.tensor(preprocessed_image, dtype=torch.float32)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            output = outputs[0]
            predicted_text = decode_prediction(output)
        return {
            'id': i+1,
            'text': predicted_text,
            'box': boxes_info[i]
        }
    try:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            segments = list(executor.map(process_line, list(enumerate(line_images))))
    except Exception as e:
        os.unlink(temp_path)
        return JsonResponse({'error': f'Line processing failed: {str(e)}'}, status=500)
    
    # Tạo ảnh để hiển thị văn bản đã parse
    try:
        # Tạo một ảnh trắng có kích thước tương tự ảnh gốc
        canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Xóa file tạm gốc
        os.unlink(temp_path)
        
        # Chuyển đổi canvas thành base64 để gửi về client
        _, buffer = cv2.imencode('.png', canvas)
        canvas_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Kết hợp kết quả của tất cả các box
        combined_text = "\n".join([segment['text'] for segment in segments])
        
        # Save original image as base64 for potential next iteration
        _, original_buffer = cv2.imencode('.jpg', original_img)
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        return JsonResponse({
            'prediction': combined_text,
            'segments': segments,
            'canvas': canvas_base64,
            'imageWidth': img_width,
            'imageHeight': img_height,
            'originalImage': original_base64,
            'currentIteration': iteration,
            'canContinue': iteration < 5  # Limit to 5 iterations
        })
    except Exception as e:
        return JsonResponse({'error': f'Canvas creation failed: {str(e)}'}, status=500)
