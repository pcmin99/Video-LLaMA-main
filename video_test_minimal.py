# video_test_minimal.py
import torch
import cv2
import einops
from video_llama.models.video_llama import VideoLLAMA
from transformers import LlamaTokenizer

# ✅ 1. 모델과 토크나이저 초기화
llama_model_path = "C:/Users/chan/Downloads/llama-7b"  # LLAMA 사전학습 모델 경로
qformer_ckpt = "C:/Users/chan/Downloads/Video-LLaMA-main/video_llama/checkpoints/qformer.pth"  # 예시 Q-Former 체크포인트

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VideoLLAMA(
    llama_model=llama_model_path,
    q_former_model=qformer_ckpt,
)
model.to(device)
model.eval()

tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)

# ✅ 2. 영상 파일 경로
video_path = "C:/Users/chan/Downloads/Video-LLaMA-main/examples/applausing.mp4"

# ✅ 3. 영상 처리 (첫 프레임만)
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("영상 파일을 읽을 수 없습니다.")
    # OpenCV는 BGR, PyTorch는 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # HWC -> CHW
    frame = einops.rearrange(frame, "h w c -> c h w")
    # float tensor & normalize
    frame = torch.tensor(frame).unsqueeze(0).float() / 255.0
    return frame

frame_tensor = get_first_frame(video_path).to(device)

# ✅ 4. Video-LLaMA 전용 인코딩
with torch.no_grad():
    video_features, _ = model.encode_videoQformer_visual(frame_tensor)

# ✅ 5. 테스트 질문 입력 (간단하게 토크나이즈)
question = "What is happening in this video?"
inputs = tokenizer(question, return_tensors="pt").to(device)

# ✅ 6. LLAMA 모델에 피쳐 + 질문 넣기 (간단 예시)
# 실제 Video-LLaMA 구조에서는 feature + input_ids embedding 결합 필요
# 여기서는 embedding 차원만 맞춰서 dummy forward
input_embeds = torch.cat([video_features, model.llama_model.model.embed_tokens(inputs.input_ids)], dim=1)

attention_mask = torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(device)

outputs = model.llama_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
logits = outputs.logits

# ✅ 7. 토큰 디코딩
pred_ids = torch.argmax(logits, dim=-1)
answer = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
print("Video Analysis Answer:", answer)
