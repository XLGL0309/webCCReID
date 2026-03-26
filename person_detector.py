import cv2
import numpy as np
import os
from PIL import Image

class PersonDetector:
    def __init__(self):
        # 加载YOLO模型
        # 使用ultralytics库的YOLOv8模型
        from ultralytics import YOLO
        # 加载预训练的YOLOv8x模型（最强大的模型）
        self.model = YOLO('yolov8x.pt')
        
    def detect_persons(self, image_path):
        """检测图片中的行人"""
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        # 使用YOLO模型检测行人，极低置信度阈值以确保检测到所有可能的行人
        results = self.model(image_path, conf=0.1, iou=0.35, imgsz=800)
        
        # 提取行人检测结果
        person_boxes = []
        img_height, img_width = img.shape[:2]
        
        # 打印调试信息
        print(f"图片尺寸: {img_width}x{img_height}")
        
        for result in results:
            boxes = result.boxes
            print(f"检测到 {len(boxes)} 个目标")
            for i, box in enumerate(boxes):
                # 检查类别是否为人（COCO数据集的类别0是person）
                if box.cls == 0:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    confidence = box.conf[0].item()
                    
                    # 打印检测到的行人信息
                    print(f"检测到行人 {i}: 位置=({x}, {y}, {w}, {h}), 置信度={confidence:.2f}")
                    
                    # 完全信任YOLO的检测结果，不做任何过滤
                    # 让相似度计算来决定哪些检测结果是相关的
                    person_boxes.append((x, y, w, h, confidence))
        
        # 按置信度排序
        person_boxes.sort(key=lambda x: x[4], reverse=True)
        
        # 打印排序后的行人信息
        print(f"排序后的行人数量: {len(person_boxes)}")
        for i, (x, y, w, h, confidence) in enumerate(person_boxes[:10]):
            print(f"行人 {i} (置信度: {confidence:.2f}): 位置=({x}, {y}, {w}, {h})")
        
        # 转换为边界框格式
        filtered_bodies = [(x, y, w, h) for x, y, w, h, confidence in person_boxes]
        
        # 如果没有检测到行人，使用整张图片
        if not filtered_bodies:
            print("没有检测到行人，使用整张图片")
            filtered_bodies = [(0, 0, img_width, img_height)]
        
        # 转换为PIL格式并返回
        detected_persons = []
        for (x, y, w, h) in filtered_bodies:
            # 确保边界框有效
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            w = x2 - x
            h = y2 - y
            
            # 裁剪行人区域
            person_img = img[y:y+h, x:x+w]
            # 转换为PIL格式
            person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            detected_persons.append(person_pil)
        
        return detected_persons

    def process_image(self, image_path, output_dir):
        """处理图片，检测并保存行人区域"""
        persons = self.detect_persons(image_path)
        output_paths = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建临时目录: {output_dir}")
        
        print(f"处理图片: {image_path}")
        print(f"检测到 {len(persons)} 个行人")
        
        for i, person in enumerate(persons):
            # 生成输出路径
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}_person_{i}{ext}")
            
            # 保存行人区域
            try:
                person.save(output_path)
                output_paths.append(output_path)
                print(f"成功保存行人图片: {output_path}")
                # 检查文件是否存在
                if os.path.exists(output_path):
                    print(f"文件存在: {output_path}, 大小: {os.path.getsize(output_path)} bytes")
                else:
                    print(f"文件不存在: {output_path}")
            except Exception as e:
                print(f"保存行人图片失败: {e}")
        
        return output_paths