from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import cv2
import numpy as np
from model import load_model, extract_features
from person_detector import PersonDetector
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine

app = Flask(__name__)
# 设置上传文件大小限制（1GB）
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 定义after_request钩子，在响应发送后删除文件
@app.after_request
def delete_files_after_request(response):
    if 'FILES_TO_DELETE' in app.config:
        files = app.config.pop('FILES_TO_DELETE')
        import threading
        import os
        
        def delete_files():
            import time
            # 根据图片数量动态计算延迟时间
            # 基础延迟10秒，每张图片增加0.5秒，最大延迟60秒
            base_delay = 10
            per_image_delay = 0.5
            max_delay = 60
            
            # 计算总图片数（探针图+底库图片+检测到的行人图片）
            total_images = 1 + len(files.get('gallery', [])) + len(files.get('detected_persons', []))
            # 计算延迟时间
            delay_time = base_delay + (total_images * per_image_delay)
            # 确保延迟时间不超过最大值
            delay_time = min(delay_time, max_delay)
            
            print(f"等待 {delay_time:.1f} 秒后删除文件，共 {total_images} 张图片")
            time.sleep(delay_time)
            
            try:
                # 删除探针图
                if 'probe' in files and os.path.exists(files['probe']):
                    os.remove(files['probe'])
                # 删除底库图片
                if 'gallery' in files:
                    for gallery_path in files['gallery']:
                        if os.path.exists(gallery_path):
                            os.remove(gallery_path)
                # 删除检测到的行人图片
                if 'detected_persons' in files:
                    for person_path in files['detected_persons']:
                        if os.path.exists(person_path):
                            os.remove(person_path)
                # 删除临时目录
                temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
                if os.path.exists(temp_dir):
                    try:
                        os.rmdir(temp_dir)
                    except:
                        pass  # 目录不为空时忽略
            except Exception as e:
                print(f"删除文件时出错: {e}")
        
        # 启动线程删除文件
        threading.Thread(target=delete_files).start()
    return response

# 加载模型
model_path = 'E:/PythonProjects/hyx_AIM_CCReID/results/ltcc/1/eval_single_gpu_3060/best_model.pth.tar'
model = load_model(model_path)

# 初始化行人检测器
detector = PersonDetector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # 检查是否有探针图
    if 'probe' not in request.files:
        return redirect(request.url)
    probe = request.files['probe']
    if probe.filename == '':
        return redirect(request.url)
    
    # 检查是否有底库文件夹
    if 'gallery' not in request.files:
        return redirect(request.url)
    gallery_files = request.files.getlist('gallery')
    if not gallery_files:
        return redirect(request.url)
    
    # 创建临时目录存储检测到的行人
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # 打印调试信息
    print(f"临时目录: {temp_dir}")
    print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    
    # 保存探针图
    if probe and allowed_file(probe.filename):
        probe_filename = secure_filename(probe.filename)
        probe_path = os.path.join(app.config['UPLOAD_FOLDER'], probe_filename)
        probe.save(probe_path)
    else:
        return redirect(request.url)
    
    # 保存底库图片
    gallery_paths = []
    for file in gallery_files:
        if file and allowed_file(file.filename):
            gallery_filename = secure_filename(file.filename)
            gallery_path = os.path.join(app.config['UPLOAD_FOLDER'], gallery_filename)
            file.save(gallery_path)
            gallery_paths.append(gallery_path)
    
    # 提取探针图特征
    probe_features = extract_features(model, probe_path)
    
    # 处理底库图片：检测行人并提取特征
    gallery_features = []
    detected_person_paths = []
    
    for gallery_path in gallery_paths:
        # 检测行人
        person_paths = detector.process_image(gallery_path, temp_dir)
        
        if person_paths:
            # 如果检测到行人，对每个行人提取特征
            print(f"检测到 {len(person_paths)} 个行人 in {gallery_path}")
            for person_path in person_paths:
                print(f"保存行人图片到: {person_path}")
                # 检查文件是否存在
                if os.path.exists(person_path):
                    print(f"文件存在: {person_path}, 大小: {os.path.getsize(person_path)} bytes")
                else:
                    print(f"文件不存在: {person_path}")
                features = extract_features(model, person_path)
                # 保存原始图片路径和裁剪后的图片路径
                gallery_features.append((person_path, features, gallery_path))
                detected_person_paths.append(person_path)
        else:
            # 如果没有检测到行人，使用整张图片
            print(f"未检测到行人，使用整张图片: {gallery_path}")
            features = extract_features(model, gallery_path)
            gallery_features.append((gallery_path, features, gallery_path))
    
    # 计算相似度并排序 - 参考单图匹配代码的相似度计算方式
    similarities = []
    for person_path, features, original_path in gallery_features:
        # 计算余弦相似度 - 使用scipy的余弦距离函数
        similarity = 1 - cosine(probe_features, features)
        # 确保相似度在0~1范围内
        similarity = np.clip(similarity, 0.0, 1.0)
        similarities.append((person_path, similarity, original_path))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 准备结果
    results = []
    for person_path, similarity, original_path in similarities:
        # 获取原始图片的文件名
        original_filename = os.path.basename(original_path)
        # 计算相对路径
        # 确保使用相对路径，从uploads目录开始
        if 'uploads' in person_path:
            # 从uploads目录开始的相对路径
            relative_path = os.path.relpath(person_path, app.config['UPLOAD_FOLDER'])
            # 转换为正斜杠
            relative_path = relative_path.replace('\\', '/')
        else:
            # 如果是原始图片（未裁剪），使用文件名
            relative_path = os.path.basename(person_path)
        results.append({
            'path': relative_path,
            'similarity': float(similarity),
            'original_filename': original_filename
        })
        # 打印调试信息
        print(f"准备结果: 原始路径={person_path}, 相对路径={relative_path}")
        print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    
    # 存储需要删除的文件路径，供after_request钩子使用
    files_to_delete = {
        'probe': probe_path,
        'gallery': gallery_paths,
        'detected_persons': detected_person_paths
    }
    app.config['FILES_TO_DELETE'] = files_to_delete
    
    return render_template('result.html', probe=probe_filename, results=results)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """提供上传文件的访问，支持子目录"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)