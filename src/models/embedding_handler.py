from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingHandler:
    """텍스트 임베딩 생성을 위한 핸들러"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self._load_model()

    def _load_model(self):
        """임베딩 모델 로드"""
        try:
            logger.info(f"임베딩 모델 로딩 중: {self.model_name}")
            # 새로운 디렉토리에 캐시
            cache_dir = "/home/ubuntu_euphoria/.huggingface_models"
            # 단일 GPU 사용을 위해 device를 cuda:0으로 고정
            self.model = SentenceTransformer(self.model_name, cache_folder=cache_dir, device='cuda:0')
            
            # LangChain과 호환되는 임베딩 클래스 생성
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cuda:0'},  # 단일 GPU 고정
                cache_folder=cache_dir
            )
            
            logger.info(f"임베딩 모델 로딩 완료: {self.model_name}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {e}")
            raise

    def create_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """텍스트의 임베딩 생성"""
        try:
            if isinstance(text, str):
                embedding = self.model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
            elif isinstance(text, list):
                embeddings = self.model.encode(text, normalize_embeddings=True)
                return embeddings.tolist()
            else:
                raise ValueError("텍스트는 문자열 또는 문자열 리스트여야 합니다")
                
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 코사인 유사도 계산"""
        try:
            embedding1 = self.model.encode(text1, normalize_embeddings=True)
            embedding2 = self.model.encode(text2, normalize_embeddings=True)
            
            # 코사인 유사도 계산
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"유사도 계산 오류: {e}")
            return 0.0

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "loaded": self.model is not None
        }