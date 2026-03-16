from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import cv2
import numpy as np
from model import load_model, extract_features
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 加载模型
model_path = 'E:\\PythonProjects\\paper\\hyx_AIM_CCReID\\results\\ltcc\\1\\eval_single_gpu_3060\\best_model.pth.tar'
model = load_model(model_path)

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
    
    # 提取特征
    probe_features = extract_features(model, probe_path)
    gallery_features = []
    for gallery_path in gallery_paths:
        features = extract_features(model, gallery_path)
        gallery_features.append((gallery_path, features))
    
    # 计算相似度并排序 - 参考单图匹配代码的相似度计算方式
    similarities = []
    for gallery_path, features in gallery_features:
        # 计算余弦相似度 - 使用scipy的余弦距离函数
        similarity = 1 - cosine(probe_features, features)
        # 确保相似度在0~1范围内
        similarity = np.clip(similarity, 0.0, 1.0)
        similarities.append((gallery_path, similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 准备结果
    results = []
    for gallery_path, similarity in similarities:
        results.append({
            'path': gallery_path.replace('uploads\\', ''),
            'similarity': float(similarity)
        })
    
    return render_template('result.html', probe=probe_filename, results=results)

if __name__ == '__main__':
    app.run(debug=True)