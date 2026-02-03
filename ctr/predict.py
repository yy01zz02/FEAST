import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from model import FoodCTRModel
from dataset import ResizeWithPadding

class CTRPredictor:
    """CTR 预测器"""
    def __init__(self, 
                 checkpoint_path,
                 bert_model_name="hfl/chinese-bert-wwm-ext",
                 image_model_name="facebook/convnextv2-huge-22k-512",
                 device=None):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载 tokenizer 和 image processor
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(image_model_name)
        
        # 加载模型
        self.model = FoodCTRModel(
            bert_model_name=bert_model_name,
            image_model_name=image_model_name,
            hidden_dims=[512, 256],
            fc_hidden_size=[128],
            drop_prob=0.2,
            num_transformer_layers=4,
            num_transformer_heads=4
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.resize_with_padding = ResizeWithPadding(512, fill_color=(255, 255, 255))
        
        print(f"Model loaded from {checkpoint_path}")
        # print(f"Best MSE during training: {checkpoint.get('best_mse', 'N/A')}")
    
    def _prepare_text(self, food_name, food_type, city_name, shop_name):
        """准备文本输入"""
        text = f"[CLS] 菜品:{food_name} [SEP] 类型:{food_type} [SEP] 城市:{city_name} [SEP] 店铺:{shop_name} [SEP]"
        encoded = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']
    
    def _prepare_image(self, image_path):
        """准备图片输入（与训练时保持一致）"""
        # 1. 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 2. 使用与训练时相同的 padding resize
        image = self.resize_with_padding(image)
        
        # 3. 使用 image_processor 进行归一化等处理
        pixel_values = self.image_processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values
        
        return pixel_values
    
    def predict(self, food_name, food_type, city_name, shop_name, image_path):
        """
        预测单条数据的 CTR
        
        Args:
            food_name: 菜品名称，如 "红烧肉饭"
            food_type: 菜品类型，如 "中式快餐"
            city_name: 城市名称，如 "北京"
            shop_name: 店铺名称，如 "老王饭店"
            image_path: 图片路径，如 "./images/10001.png"
        
        Returns:
            ctr:  预测的点击率 (0-1之间的浮点数)
        """
        with torch.no_grad():
            # 准备输入
            input_ids, attention_mask = self._prepare_text(food_name, food_type, city_name, shop_name)
            pixel_values = self._prepare_image(image_path)
            
            # 移动到设备
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values = pixel_values.to(self.device)
            
            # 预测
            ctr_pred, text_emb, img_emb = self.model(input_ids, attention_mask, pixel_values)
            
            return ctr_pred.item()
    