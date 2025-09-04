import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import logging
from typing import Dict, List, Optional, Any, Iterator
import threading
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    """3개 모델만 지원하는 LLM 핸들러"""
    
    # 지원하는 3개 모델만 정의
    SUPPORTED_MODELS = {
        "qwen2.5-7b": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "description": "Qwen 2.5 7B - 고성능 범용 모델",
            "category": "medium",
            "ram_requirement": "16GB",
            "gpu_requirement": "8GB",
            "performance_score": 85,
            "use_cases": ["general", "korean", "coding"]
        },
        "llama3.1-8b": {
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "description": "Meta Llama 3 8B - 고성능 모델",
            "category": "medium", 
            "ram_requirement": "16GB",
            "gpu_requirement": "8GB",
            "performance_score": 88,
            "use_cases": ["general", "coding", "reasoning"]
        },
        "gemma-3-4b": {
            "model_id": "google/gemma-2-9b-it",
            "description": "Google Gemma 2 9B - 효율적인 중형 모델",
            "category": "medium",
            "ram_requirement": "18GB", 
            "gpu_requirement": "10GB",
            "performance_score": 82,
            "use_cases": ["general", "multilingual"]
        }
    }

    def __init__(self, model_key: str = "llama3.1-8b"):
        self.model_key = model_key
        self.model = None
        self.tokenizer = None
        self.chat_model = None
        
        if model_key not in self.SUPPORTED_MODELS:
            raise ValueError(f"지원되지 않는 모델: {model_key}. 지원 모델: {list(self.SUPPORTED_MODELS.keys())}")
            
        self._load_model()

    def _load_model(self):
        """모델 로드"""
        try:
            model_info = self.SUPPORTED_MODELS[self.model_key]
            model_id = model_info["model_id"]
            
            logger.info(f"모델 로딩 중: {model_id}")
            
            # 양자화 설정 (GPU 메모리 절약)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # 토크나이저 로드 (새로운 디렉토리에 캐시)
            cache_dir = "/home/ubuntu_euphoria/.huggingface_models"
            
            # Hugging Face 토큰 설정
            import os
            from dotenv import load_dotenv
            
            # .env 파일 로드
            load_dotenv()
            
            token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
            
            # 토큰이 없으면 직접 입력하도록 안내
            if not token:
                print("❌ Hugging Face 토큰이 설정되지 않았습니다.")
                print("해결 방법:")
                print("1. https://huggingface.co/settings/tokens 에서 토큰 생성")
                print("2. .env 파일에 HUGGINGFACE_TOKEN='your_token' 추가")
                print("3. 또는 환경변수 설정: set HUGGINGFACE_TOKEN=your_token")
                raise ValueError("Hugging Face 토큰이 필요합니다.")
            
            logger.info(f"Hugging Face 토큰 확인됨: {token[:10]}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                token=token
            )
            # pad_token을 별도로 설정하여 attention mask 경고 해결
            if self.tokenizer.pad_token is None:
                # Llama 모델의 경우 eos_token을 pad_token으로 사용
                if self.model_key.startswith("llama"):
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "[PAD]" if "[PAD]" in self.tokenizer.vocab else self.tokenizer.unk_token
            
            # 모델 로드 (새로운 디렉토리에 캐시)
            # 단일 GPU 사용을 위해 device_map을 cuda:0으로 고정
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="cuda:0",  # 단일 GPU 사용
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir=cache_dir,
                token=token
            )
            
            # 채팅을 위한 파이프라인 설정
            from transformers import pipeline
            self.chat_model = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            logger.info(f"모델 로딩 완료: {model_id}")
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 512, stream: bool = False):
        """텍스트 생성 (스트리밍 지원)"""
        if stream:
            return self._generate_stream(prompt, max_length)
        else:
            return self._generate_normal(prompt, max_length)
    
    def _generate_normal(self, prompt: str, max_length: int) -> str:
        """일반 텍스트 생성"""
        try:
            # 모델별 프롬프트 포맷팅
            formatted_prompt = self._format_prompt(prompt)
            
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 터미널에 생성된 답변 출력
            logger.info(f"생성된 답변: {response.strip()[:100]}{'...' if len(response.strip()) > 100 else ''}")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {e}")
            return f"오류 발생: {str(e)}"
    
    def _generate_stream(self, prompt: str, max_length: int) -> Iterator[str]:
        """스트리밍 텍스트 생성"""
        try:
            # 모델별 프롬프트 포맷팅
            formatted_prompt = self._format_prompt(prompt)
            
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)
            
            # TextIteratorStreamer 설정
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                timeout=60, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                'inputs': inputs.input_ids,
                'attention_mask': inputs.attention_mask,
                'max_new_tokens': max_length,
                'temperature': 0.7,
                'do_sample': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer,
            }
            
            # 백그라운드에서 생성 실행
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 스트리밍 토큰 반환 및 터미널 출력
            full_response = ""
            first_token = True
            for text in streamer:
                if text:
                    if first_token:
                        try:
                            print("[스트리밍] ", end="", flush=True)
                        except UnicodeEncodeError:
                            pass  # 인코딩 오류 시 터미널 출력 생략
                        first_token = False
                    full_response += text
                    try:
                        print(text, end="", flush=True)  # 터미널에 실시간 출력
                    except UnicodeEncodeError:
                        pass  # 인코딩 오류 시 터미널 출력 생략
                    yield text  # JSON 형식이 아닌 단순 텍스트로 변경
            
            try:
                print()  # 줄바꿈
            except UnicodeEncodeError:
                pass
            logger.info(f"스트리밍 생성 완료: {len(full_response)}자")
            
            # 완료 신호는 더 이상 필요 없음 (단순 텍스트 스트리밍)
            
        except Exception as e:
            logger.error(f"스트리밍 생성 오류: {e}")
            # 오류 시에도 단순 텍스트 반환
            yield f"오류 발생: {str(e)}"

    def chat_generate(self, message: str, stream: bool = False):
        """채팅 형태 응답 생성 (스트리밍 지원)"""
        # chat_generate는 generate 메서드를 사용하도록 변경
        return self.generate(message, max_length=512, stream=stream)

    def _format_prompt(self, prompt: str) -> str:
        """모델별 프롬프트 포맷팅"""
        if self.model_key.startswith("qwen"):
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif self.model_key.startswith("llama"):
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:  # gemma
            return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _format_chat_prompt(self, message: str) -> str:
        """채팅용 프롬프트 포맷팅"""
        return self._format_prompt(message)

    def get_model_info(self) -> Dict[str, Any]:
        """현재 모델 정보 반환"""
        return {
            "model_key": self.model_key,
            "model_id": self.SUPPORTED_MODELS[self.model_key]["model_id"],
            "description": self.SUPPORTED_MODELS[self.model_key]["description"],
            "category": self.SUPPORTED_MODELS[self.model_key]["category"],
            "loaded": self.model is not None
        }

    @classmethod
    def get_supported_models(cls) -> Dict[str, Dict[str, Any]]:
        """지원되는 모델 목록 반환"""
        return cls.SUPPORTED_MODELS

    @classmethod
    def get_model_categories(cls) -> List[str]:
        """모델 카테고리 목록 반환"""
        categories = set()
        for model in cls.SUPPORTED_MODELS.values():
            categories.add(model["category"])
        return list(categories)

    @classmethod
    def get_models_by_category(cls, category: str = None) -> Dict[str, Dict[str, Any]]:
        """카테고리별 모델 반환"""
        if not category:
            # 모든 카테고리별로 그룹화
            result = {}
            for key, model in cls.SUPPORTED_MODELS.items():
                cat = model["category"]
                if cat not in result:
                    result[cat] = {}
                result[cat][key] = model
            return result
        else:
            # 특정 카테고리만
            return {k: v for k, v in cls.SUPPORTED_MODELS.items() if v["category"] == category}

    @classmethod
    def recommend_model(cls, ram_gb: int = None, gpu_gb: int = None, use_case: str = None) -> List[Dict[str, Any]]:
        """시스템 사양과 용도에 맞는 모델 추천"""
        recommendations = []
        
        for key, model in cls.SUPPORTED_MODELS.items():
            score = 0
            reasons = []
            
            # RAM 요구사항 체크
            model_ram = int(model["ram_requirement"].replace("GB", ""))
            if ram_gb and ram_gb >= model_ram:
                score += 30
                reasons.append(f"RAM 요구사항 충족 ({model_ram}GB)")
            elif ram_gb:
                score -= 20
                reasons.append(f"RAM 부족 (필요: {model_ram}GB, 보유: {ram_gb}GB)")
            
            # GPU 요구사항 체크  
            model_gpu = int(model["gpu_requirement"].replace("GB", ""))
            if gpu_gb and gpu_gb >= model_gpu:
                score += 30
                reasons.append(f"GPU 메모리 요구사항 충족 ({model_gpu}GB)")
            elif gpu_gb:
                score -= 20
                reasons.append(f"GPU 메모리 부족 (필요: {model_gpu}GB, 보유: {gpu_gb}GB)")
            
            # 용도별 점수
            if use_case and use_case in model["use_cases"]:
                score += 25
                reasons.append(f"'{use_case}' 용도에 적합")
            
            # 기본 성능 점수
            score += model["performance_score"] // 10
            
            recommendations.append({
                "model_key": key,
                "model_info": model,
                "recommendation_score": max(0, score),
                "reasons": reasons
            })
        
        # 점수순으로 정렬
        recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
        return recommendations

    @classmethod
    def get_performance_comparison(cls, model_keys: List[str] = None) -> List[Dict[str, Any]]:
        """모델 성능 비교"""
        if not model_keys:
            model_keys = list(cls.SUPPORTED_MODELS.keys())
        
        comparison = []
        for key in model_keys:
            if key in cls.SUPPORTED_MODELS:
                model = cls.SUPPORTED_MODELS[key]
                comparison.append({
                    "model_key": key,
                    "performance_score": model["performance_score"],
                    "ram_requirement": model["ram_requirement"],
                    "gpu_requirement": model["gpu_requirement"],
                    "category": model["category"],
                    "description": model["description"]
                })
        
        # 성능 점수순으로 정렬
        comparison.sort(key=lambda x: x["performance_score"], reverse=True)
        return comparison

    @classmethod
    def get_model_info(cls, model_key: str) -> Optional[Dict[str, Any]]:
        """특정 모델 정보 반환"""
        return cls.SUPPORTED_MODELS.get(model_key)