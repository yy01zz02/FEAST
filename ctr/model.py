import torch
import torch.nn as nn
from transformers import (
    BertConfig, BertModel, AutoModel, AutoTokenizer,
    AutoImageProcessor, ConvNextV2Model
)

class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name="facebook/convnextv2-huge-22k-512",
                 hidden_dims=[512, 256],
                 drop_prob=0.2,
                 freeze_backbone=False):
        super().__init__()
        
        self.backbone = ConvNextV2Model.from_pretrained(model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        
        # ConvNeXt V2 output dimension
        self.backbone_out_size = self.backbone.config.hidden_sizes[-1]
        print(f"ConvNeXt V2 output size: {self.backbone_out_size}")
        
        if freeze_backbone: 
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Image backbone frozen")
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection layers with LayerNorm (more stable than BatchNorm for small batches)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_out_size, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(p=drop_prob)
        )

    
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        features = outputs.last_hidden_state  # [B, C, H, W]
        pooled = self.global_pool(features).flatten(1)  # [B, C]
        img_emb = self.projector(pooled)
        return img_emb


class TextEncoder(nn.Module):
    def __init__(self,
                 bert_model_name="hfl/chinese-bert-wwm-ext",
                 hidden_dims=[512, 256],
                 drop_prob=0.2,
                 freeze_bert=False):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_out_size = self.bert.config.hidden_size
        print(f"BERT output size: {bert_out_size}")
        
        if freeze_bert: 
            for param in self.bert.parameters():
                param.requires_grad = False
            print("BERT frozen")

        # Projection layers with LayerNorm
        self.projector = nn.Sequential(
            nn.Linear(bert_out_size, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(p=drop_prob)
        )
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_output.last_hidden_state[:, 0, :]
        text_emb = self.projector(cls_emb)
        return text_emb


class CrossModalTransformer(nn.Module):
    def __init__(self, hidden_size=256, num_layers=4, num_heads=4, drop_prob=0.1):
        super().__init__()
        
        config = BertConfig()
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_layers
        config.num_attention_heads = num_heads
        config.intermediate_size = hidden_size * 4
        config.hidden_dropout_prob = drop_prob
        config.attention_probs_dropout_prob = drop_prob
        config.max_position_embeddings = 8
        config.vocab_size = 1
        
        self.transformer = BertModel(config)
        self.modality_embeddings = nn.Embedding(2, hidden_size)
        
    def forward(self, text_emb, img_emb):
        batch_size = text_emb.size(0)
        device = text_emb.device
        
        text_type = torch.zeros(batch_size, dtype=torch.long, device=device)
        img_type = torch.ones(batch_size, dtype=torch.long, device=device)
        
        text_emb = text_emb + self.modality_embeddings(text_type)
        img_emb = img_emb + self.modality_embeddings(img_type)
        
        multi_emb = torch.stack([text_emb, img_emb], dim=1)
        attention_mask = torch.ones(batch_size, 2, device=device)
        
        encoded_output = self.transformer.encoder(
            multi_emb,
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
        ).last_hidden_state
        
        return encoded_output[: , 0, : ], encoded_output[:, 1, :]


class FoodCTRModel(nn.Module):
    def __init__(self,
                 bert_model_name="hfl/chinese-bert-wwm-ext",
                 image_model_name="facebook/convnextv2-huge-22k-512",
                 hidden_dims=[512, 256],
                 fc_hidden_size=[128],
                 drop_prob=0.2,
                 num_transformer_layers=4,
                 num_transformer_heads=4,
                 freeze_bert=False,
                 freeze_image_backbone=False):
        super().__init__()
        
        self.text_encoder = TextEncoder(
            bert_model_name=bert_model_name,
            hidden_dims=hidden_dims,
            drop_prob=drop_prob,
            freeze_bert=freeze_bert
        )
        
        self.image_encoder = ImageEncoder(
            model_name=image_model_name,
            hidden_dims=hidden_dims,
            drop_prob=drop_prob,
            freeze_backbone=freeze_image_backbone
        )
        
        self.cross_modal_transformer = CrossModalTransformer(
            hidden_size=hidden_dims[-1],
            num_layers=num_transformer_layers,
            num_heads=num_transformer_heads,
            drop_prob=drop_prob
        )
        
        final_dim = hidden_dims[-1] * 2
        
        # Final fully connected layer
        self.final_fc = nn.Sequential(
            nn.Linear(final_dim, fc_hidden_size[0]),
            nn.LayerNorm(fc_hidden_size[0]),  
            nn.GELU(),
            nn.Dropout(p=drop_prob),
        )
        
        # Prediction head: outputs unnormalized logits (no sigmoid)
        # This allows using the Distribution-aware Compound Loss
        self.ctr_head = nn.Linear(fc_hidden_size[-1], 1)
        
        # Initialize the final layer to avoid initial bias
        nn.init.xavier_uniform_(self.ctr_head.weight)
        nn.init.zeros_(self.ctr_head.bias)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        text_emb = self.text_encoder(input_ids, attention_mask)
        img_emb = self.image_encoder(pixel_values)
        
        fused_text, fused_img = self.cross_modal_transformer(text_emb, img_emb)
        
        fused_emb = torch.cat([fused_text, fused_img], dim=-1)
        hidden = self.final_fc(fused_emb)
        ctr_pred = self.ctr_head(hidden)
        
        return ctr_pred, text_emb, img_emb