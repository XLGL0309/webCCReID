from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import cv2
import numpy as np
from model import load_model, extract_features
from person_detector import PersonDetector
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine
import tempfile

app = Flask(__name__)
# 设置上传文件大小限制（1GB）
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'wmv'}

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
            # 基础延迟30秒，每张图片增加1秒，最大延迟120秒
            # 增加延迟时间以确保所有处理完成
            base_delay = 30
            per_image_delay = 1.0
            max_delay = 120
            
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
                    try:
                        os.remove(files['probe'])
                        print(f"删除探针图: {files['probe']}")
                    except Exception as e:
                        print(f"删除探针图失败: {e}")
                
                # 删除底库图片
                if 'gallery' in files:
                    for gallery_path in files['gallery']:
                        if os.path.exists(gallery_path):
                            try:
                                os.remove(gallery_path)
                                print(f"删除底库图片: {gallery_path}")
                            except Exception as e:
                                print(f"删除底库图片失败: {e}")
                
                # 删除检测到的行人图片
                if 'detected_persons' in files:
                    for person_path in files['detected_persons']:
                        if os.path.exists(person_path):
                            try:
                                os.remove(person_path)
                                print(f"删除检测到的行人图片: {person_path}")
                            except Exception as e:
                                print(f"删除行人图片失败: {e}")
                
                # 删除临时目录
                temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
                if os.path.exists(temp_dir):
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        print(f"删除临时目录: {temp_dir}")
                    except Exception as e:
                        print(f"删除临时目录失败: {e}")
                
                # 删除视频帧临时目录
                video_temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'video_frames')
                if os.path.exists(video_temp_dir):
                    try:
                        import shutil
                        shutil.rmtree(video_temp_dir)
                        print(f"删除视频帧临时目录: {video_temp_dir}")
                    except Exception as e:
                        print(f"删除视频帧目录时出错: {e}")
            except Exception as e:
                print(f"删除文件时出错: {e}")
        
        # 启动线程删除文件
        threading.Thread(target=delete_files).start()
    return response

# 获取所有可用的模型
def get_available_models():
    """获取results目录下所有可用的模型"""
    models = []
    results_path = 'E:/PythonProjects/hyx_AIM_CCReID/results'
    
    if os.path.exists(results_path):
        # 遍历数据集目录（ltcc, prcc等）
        for dataset in os.listdir(results_path):
            dataset_path = os.path.join(results_path, dataset)
            if os.path.isdir(dataset_path):
                # 遍历训练次数目录（数字）
                for train_num in os.listdir(dataset_path):
                    train_path = os.path.join(dataset_path, train_num)
                    if os.path.isdir(train_path):
                        # 查找模型文件
                        model_path = os.path.join(train_path, 'eval_single_gpu_3060', 'best_model.pth.tar')
                        if os.path.exists(model_path):
                            model_name = f"{dataset}_{train_num}"
                            models.append({'name': model_name, 'path': model_path})
    
    return models

# 初始化行人检测器
detector = PersonDetector()

# 全局模型变量
current_model = None
current_model_path = None

# 视频帧提取函数
def extract_video_frames(video_path, output_dir, frames_per_second=3):
    """从视频中提取帧，每秒提取指定数量的帧"""
    frames = []
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return frames
    
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算每帧之间的间隔
    frame_interval = int(fps / frames_per_second)
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按照指定的间隔提取帧
        if frame_count % frame_interval == 0:
            # 生成帧的文件名
            frame_filename = f"frame_{extracted_count}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # 保存帧
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"从视频中提取了 {extracted_count} 帧")
    return frames

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # 获取所有可用的模型
    models = get_available_models()
    return render_template('index.html', models=models)

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
    
    # 检查是否选择了模型
    if 'model' not in request.form:
        return redirect(request.url)
    model_path = request.form['model']
    
    # 加载选择的模型
    global current_model, current_model_path
    if current_model_path != model_path:
        print(f"加载新模型: {model_path}")
        current_model = load_model(model_path)
        current_model_path = model_path
    
    # 创建临时目录存储检测到的行人
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # 打印调试信息
    print(f"临时目录: {temp_dir}")
    print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    
    # 生成唯一的文件名，避免冲突
    import uuid
    
    # 保存探针图
    if probe and allowed_file(probe.filename):
        # 生成唯一文件名
        unique_id = str(uuid.uuid4())[:8]
        probe_ext = os.path.splitext(probe.filename)[1]
        probe_filename = f"probe_{unique_id}{probe_ext}"
        probe_path = os.path.join(app.config['UPLOAD_FOLDER'], probe_filename)
        probe.save(probe_path)
    else:
        return redirect(request.url)
    
    # 保存底库图片和视频
    gallery_items = []  # 存储(文件路径, 原始文件名)元组
    video_frames = []  # 存储从视频中提取的帧路径
    
    for file in gallery_files:
        if file and allowed_file(file.filename):
            # 生成唯一文件名
            unique_id = str(uuid.uuid4())[:8]
            gallery_ext = os.path.splitext(file.filename)[1]
            gallery_filename = f"gallery_{unique_id}{gallery_ext}"
            gallery_path = os.path.join(app.config['UPLOAD_FOLDER'], gallery_filename)
            file.save(gallery_path)
            
            # 检查是否是视频文件
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            if file_ext in ['mp4', 'avi', 'mov', 'wmv']:
                print(f"处理视频文件: {gallery_path}")
                # 创建临时目录用于存储提取的帧
                video_temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'video_frames', os.path.splitext(gallery_filename)[0])
                if not os.path.exists(video_temp_dir):
                    os.makedirs(video_temp_dir, exist_ok=True)
                
                # 提取视频帧
                frames = extract_video_frames(gallery_path, video_temp_dir, frames_per_second=3)
                # 为每个视频帧保存原始视频文件名
                for frame_path in frames:
                    video_frames.append((frame_path, file.filename))
            else:
                # 图片文件直接添加，保存原始文件名
                gallery_items.append((gallery_path, file.filename))
    
    # 合并底库图片和视频帧
    gallery_items.extend(video_frames)
    
    # 提取探针图特征
    probe_features = extract_features(current_model, probe_path)
    
    # 处理底库图片：检测行人并提取特征
    gallery_features = []
    detected_person_paths = []
    
    for gallery_path, original_filename in gallery_items:
        try:
            # 检测行人
            person_paths = detector.process_image(gallery_path, temp_dir)
            
            if person_paths:
                # 如果检测到行人，对每个行人提取特征
                print(f"检测到 {len(person_paths)} 个行人 in {gallery_path}")
                for person_path in person_paths:
                    try:
                        print(f"保存行人图片到: {person_path}")
                        # 检查文件是否存在
                        if os.path.exists(person_path):
                            print(f"文件存在: {person_path}, 大小: {os.path.getsize(person_path)} bytes")
                        else:
                            print(f"文件不存在: {person_path}")
                            continue
                        
                        features = extract_features(current_model, person_path)
                        # 保存原始图片路径、裁剪后的图片路径和原始文件名
                        gallery_features.append((person_path, features, gallery_path, original_filename))
                        detected_person_paths.append(person_path)
                    except Exception as e:
                        print(f"处理行人图片失败 {person_path}: {e}")
                        continue
            else:
                # 如果没有检测到行人，使用整张图片
                print(f"未检测到行人，使用整张图片: {gallery_path}")
                features = extract_features(current_model, gallery_path)
                gallery_features.append((gallery_path, features, gallery_path, original_filename))
        except Exception as e:
            print(f"处理底库文件失败 {gallery_path}: {e}")
            continue
    
    # 计算相似度并排序 - 参考单图匹配代码的相似度计算方式
    similarities = []
    for person_path, features, original_path, original_filename in gallery_features:
        # 计算余弦相似度 - 使用scipy的余弦距离函数
        similarity = 1 - cosine(probe_features, features)
        # 确保相似度在0~1范围内
        similarity = np.clip(similarity, 0.0, 1.0)
        similarities.append((person_path, similarity, original_path, original_filename))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 准备结果
    results = []
    for person_path, similarity, original_path, original_filename in similarities:
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
        print(f"准备结果: 原始路径={person_path}, 相对路径={relative_path}, 原始文件名={original_filename}")
        print(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    
    # 存储需要删除的文件路径，供after_request钩子使用
    # 从gallery_items中提取文件路径
    gallery_paths = [item[0] for item in gallery_items]
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