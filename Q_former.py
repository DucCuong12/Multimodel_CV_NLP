import timm  
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
class Q_former_fusion(nn.Module):
    def __init__(self, num_query_tokens = 32, hidden_dim = 768):
        super(Q_former_fusion, self).__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_dim = hidden_dim
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim))  # [32, 768]
        self.img_encoder = ImageEncoder()    #ViT: size: Batch x 197 x 768
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.text_encoder = TextEncoder()     #Bert: size: Batch x 1x 768
        self.mix = nn.Linear(1536,512)
        self.fc_layer = nn.Linear(512,256)
        self.final_fc = nn.Linear(256,4)
        self.dropout = nn.Dropout(0.2)
        self.cross_Q = Cross_MHA(768,8)
        self.relu = nn.ReLU()
        
    def forward(self,img, input_ids, attention_mask):
        img = self.img_encoder(img)   #VIT: batch x 197 x 768
        B = img.size(0)
        query = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)
        cross_vit, _ = self.cross_Q(query = query, key = img, value = img)   # size: Batch x 32 x 768
        img = cross_vit.mean(dim=1)
        text = self.text_encoder(input_ids, attention_mask)

        combined = torch.cat((text, img), dim=1)  # [batch, 1024]
        combined = self.relu(self.mix(self.dropout(combined)))   # 512                
        # Feed Forward Layer
        combined = self.relu(self.fc_layer(combined))
        output = self.final_fc(combined)
        return output

# python -c "import torch; print(torch.__version__)"
# y