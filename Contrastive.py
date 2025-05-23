import timm  # Make sure to install timm: pip install timm
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer



class ImageEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224",return_all_tokens=True):
        super(ImageEncoder, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        
        # Optional: Remove classification head
        self.vit.head = nn.Identity()
        
        # Choose what to return
        self.return_all_tokens = return_all_tokens

    def forward(self, x):
        # x: [B, 3, 224, 224]
        if self.return_all_tokens:
            # Get all 197 tokens: [CLS] + 196 patch tokens
            tokens = self.vit.forward_features(x)  # Shape: [B, 197, 768]
            return tokens
        else:
            # Return only CLS token: [B, 768]
            cls_token = self.vit(x)
            return cls_token



class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
    def forward(self, input_ids, attention_mask):
        # tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        return output.last_hidden_state[:, 0, :]
    

class Cross_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Cross_MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
    def forward(self, query, key, value, key_padding_mask = None):
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask)
        return attn_output, attn_weights
class AddNorm(nn.Module):
    def __init__(self, embed_dim, eps = 1e-6):
        super(AddNorm, self).__init__() 
        self.embed_dim = embed_dim 
        self.norm = nn.LayerNorm(embed_dim, eps = eps)
    def forward(self, x, sublayer_output):
        return self.norm(x+sublayer_output)
class Q_cons_fusion(nn.Module):
    def __init__(self, num_query_tokens = 32, hidden_dim = 768):
        super(Q_cons_fusion, self).__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_dim = hidden_dim
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim))  # [32, 768]
        self.q_norm = nn.LayerNorm(hidden_dim,  eps = 1e-6)
        self.img_encoder = ImageEncoder()    #ViT: size: Batch x 197 x 768
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.text_encoder = TextEncoder()     #Bert: size: Batch x 1x 768
        # self.mix = nn.Linear(1536,512)
        # self.fc_layer = nn.Linear(512,256)
        # self.final_fc = nn.Linear(256,4)
        # self.dropout = nn.Dropout(0.2)
        self.cross_Q = Cross_MHA(768,8)
        # self.relu = nn.ReLU()
        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim, eps = 1e-6),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            # nn.LayerNorm(hidden_dim * 2, eps =1e-6),
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout to prevent early overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 30)
        )
    def forward(self,img, input_ids, attention_mask):
        img = self.img_encoder(img)   #VIT: batch x 197 x 768
        B = img.size(0)
        query = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)
        cross_vit, _ = self.cross_Q(query = query, key = img, value = img)   # size: Batch x 32 x 768
        cross_vit = self.q_norm(self.q_mlp(cross_vit)+query)
        img = cross_vit.mean(dim=1)
        text = self.text_encoder(input_ids, attention_mask)    # batch x 768
        img = F.normalize(img, dim =-1, eps=1e-6)
        text = F.normalize(text, dim=-1, eps=1e-6)
        logit_img = img @ text.T
        logit_text = text @ img.T
        combined = torch.cat((text, img), dim=1)  # [batch, 1024]
        output = self.fusion(combined)
        return output, logit_img, logit_text

def compute_itc_loss(img_feat, text_feat, temperature=0.07):
    """
    img_feat: Tensor [B, D]
    text_feat: Tensor [B, D]
    """
    # Normalize embeddings
    img_feat = F.normalize(img_feat, dim=-1, eps=1e-6)
    text_feat = F.normalize(text_feat, dim=-1, eps=1e-6)

    # Cosine similarity: [B, B]
    logits_per_image = img_feat @ text_feat.T
    logits_per_text = text_feat @ img_feat.T

    # Apply temperature
    logits_per_image /= temperature
    logits_per_text /= temperature

    # Ground-truth: diagonal (same index)
    targets = torch.arange(img_feat.size(0)).to(img_feat.device)

    # Cross entropy
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    
    # Final ITC loss
    loss = (loss_i2t + loss_t2i) / 2
    return loss