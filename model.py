import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from scipy.spatial.distance import cosine

# 设置设备 - 优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建项目路径
project_path = os.path.join(current_dir, '..', 'hyx_AIM_CCReID')
project_path = os.path.abspath(project_path)

# 添加项目路径到系统路径
sys.path.append(project_path)

# 确保路径存在
if not os.path.exists(project_path):
    print(f"警告: 项目路径 {project_path} 不存在")
    print(f"当前目录: {current_dir}")

# 导入模型相关模块
try:
    from models.img_resnet import ResNet50
    from configs.default_img import get_img_config
    import argparse
    print("成功导入模型模块")
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"项目路径: {project_path}")
    print(f"Python路径: {sys.path}")
    sys.exit(1)

# 创建默认参数
class Args:
    def __init__(self):
        self.cfg = None
        self.root = None
        self.output = None
        self.resume = None
        self.eval = False
        self.tag = None
        self.dataset = 'ltcc'
        self.gpu = '0'

# 获取配置
args = Args()
config = get_img_config(args)
print(f"配置加载成功，特征维度: {config.MODEL.FEATURE_DIM}")

def load_model(model_path):
    """加载训练好的模型 - 直接使用ResNet50模型"""
    # 创建ResNet50模型实例
    model = ResNet50(config)
    
    # 加载模型权重 - 参考单图匹配代码的权重加载方式
    print(f"[INFO] 加载模型权重：{model_path}")
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False  # 关键修改：改为False，兼容包含numpy的权重
    )
    
    # 权重加载容错处理 - 参考单图匹配代码
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[INFO] 使用model_state_dict加载权重成功")
    except KeyError:
        # 兼容直接保存model.state_dict()的情况
        try:
            model.load_state_dict(checkpoint)
            print("[INFO] 直接加载权重成功")
        except Exception as e:
            print(f"[ERROR] 权重加载失败: {e}")
            raise
    
    # 设置为评估模式
    model.eval()
    
    # 将模型移动到GPU
    model = model.to(device)
    print(f"[INFO] 模型已移动到设备: {device}")
    
    return model

def extract_features(model, image_input):
    """提取图像特征 - 参考单图匹配代码，包含水平翻转增强"""
    # 图片预处理
    data_transforms = transforms.Compose([
        transforms.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        transforms.Grayscale(num_output_channels=3),  # 转换为灰度图但保持3通道
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 处理图片
    try:
        if isinstance(image_input, str):
            # 如果输入是文件路径
            if not os.path.exists(image_input):
                print(f"图片文件不存在：{image_input}")
                raise FileNotFoundError(f"图片文件不存在：{image_input}")
            image = Image.open(image_input).convert('RGB')
        else:
            # 如果输入是PIL图像
            image = image_input.convert('RGB')
        
        image_tensor = data_transforms(image)
        input_batch = image_tensor.unsqueeze(0)
        
        # 将输入数据移动到GPU
        input_batch = input_batch.to(device)
        
        # 水平翻转增强提取特征（和训练逻辑一致）
        flip_img = torch.flip(input_batch, [3])
        flip_img = flip_img.to(device)
        
        # 提取特征 - ResNet50返回(old_x, f)
        with torch.no_grad():
            _, batch_features = model(input_batch)
            _, batch_features_flip = model(flip_img)
        
        # 融合翻转特征并归一化
        batch_features = (batch_features + batch_features_flip) / 2
        batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
        
        # 转换为numpy数组
        features = batch_features.cpu().numpy().flatten()
        return features
    except Exception as e:
        print(f"提取特征失败: {e}")
        # 返回一个全零的特征向量，长度与模型输出一致
        return np.zeros(config.MODEL.FEATURE_DIM)