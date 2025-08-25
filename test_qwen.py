import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Simple Qwen 2.5 model test without Gradio"""
    
    try:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        logger.info(f"모델 로딩 중: {model_id}")
        
        # 양자화 설정 (GPU 메모리 절약)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 토크나이저 로드
        cache_dir = "C:\\huggingface_models"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        
        # pad_token 설정하여 attention mask 경고 해결
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "[PAD]" if "[PAD]" in tokenizer.vocab else tokenizer.unk_token
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        logger.info(f"모델 로딩 완료: {model_id}")
        
        # 테스트 프롬프트
        test_prompts = [
            "안녕하세요! 자기소개를 해주세요.",
            "Python에서 리스트와 튜플의 차이점을 설명해주세요.",
            "오늘 날씨가 좋네요. 산책하기 어때요?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*50}")
            print(f"테스트 {i}: {prompt}")
            print(f"{'='*50}")
            
            # Qwen 포맷으로 프롬프트 포맷팅
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 토크나이징 (attention mask 포함)
            inputs = tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # 생성
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 응답 디코딩 
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            print(f"응답: {response}")
        
        print(f"\n{'='*50}")
        print("모든 테스트 완료!")
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"오류: {e}")

if __name__ == "__main__":
    main()