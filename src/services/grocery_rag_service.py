"""
식료품 가격 정보를 RAG로 관리하는 서비스
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config_loader import config

logger = logging.getLogger(__name__)


class GroceryRAGService:
    """식료품 가격 정보 RAG 서비스"""

    def __init__(self, rag_service):
        """
        Args:
            rag_service: 기존 RAGService 인스턴스
        """
        self.rag_service = rag_service
        self.grocery_data_path = Path(__file__).parent.parent.parent / "data" / "grocery_deals.json"

    def load_grocery_data(self) -> List[Dict[str, Any]]:
        """JSON 파일에서 식료품 데이터 로드"""
        try:
            with open(self.grocery_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"식료품 데이터 로드 완료: {len(data)}개 항목")
            return data
        except Exception as e:
            logger.error(f"식료품 데이터 로드 실패: {e}")
            return []

    def format_grocery_item(self, item: Dict[str, Any]) -> str:
        """식료품 항목을 텍스트로 포맷팅"""
        product = item['product']
        unit = item['unit']
        category = item['category']

        # 가격 정보를 정렬 (가격 낮은 순)
        stores = sorted(item['stores'], key=lambda x: x['price'])

        # 텍스트 생성
        text = f"카테고리: {category}\n"
        text += f"상품명: {product} ({unit})\n\n"
        text += "가격 비교:\n"

        for idx, store in enumerate(stores, 1):
            price = f"{store['price']:,}원"
            location = store['location']
            discount = f" - {store['discount']}" if store.get('discount') else ""
            text += f"{idx}. {store['name']}: {price} ({location}){discount}\n"

        # 최저가 정보 강조
        best_store = stores[0]
        text += f"\n최저가: {best_store['name']} - {best_store['price']:,}원"

        return text

    def create_documents_from_grocery_data(self) -> List[Document]:
        """식료품 데이터를 LangChain Document로 변환"""
        grocery_items = self.load_grocery_data()
        documents = []

        for item in grocery_items:
            text = self.format_grocery_item(item)

            # 메타데이터 생성
            metadata = {
                "source": "grocery_deals",
                "category": item['category'],
                "product": item['product'],
                "unit": item['unit'],
                "min_price": min(store['price'] for store in item['stores']),
                "max_price": max(store['price'] for store in item['stores'])
            }

            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        logger.info(f"{len(documents)}개의 식료품 문서 생성 완료")
        return documents

    def initialize_grocery_rag(self):
        """식료품 데이터를 RAG에 초기화"""
        try:
            documents = self.create_documents_from_grocery_data()

            if not documents:
                logger.warning("초기화할 식료품 데이터가 없습니다")
                return False

            # 텍스트 분할 (각 상품이 이미 충분히 작으므로 큰 chunk_size 사용)
            external_web_config = config.external_web_rag_config
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=external_web_config['chunk_size'],
                chunk_overlap=external_web_config['chunk_overlap']
            )
            chunks = text_splitter.split_documents(documents)

            # Vector DB에 추가
            self.rag_service.db.add_documents(chunks)
            logger.info(f"{len(chunks)}개의 식료품 청크를 RAG에 추가했습니다")

            return True
        except Exception as e:
            logger.error(f"식료품 RAG 초기화 실패: {e}")
            return False

    def search_grocery_prices(self, product_query: str, limit: int = 3) -> str:
        """식료품 가격 정보 검색"""
        try:
            # RAG retriever를 사용하여 관련 문서 검색
            docs = self.rag_service.retriever.invoke(product_query, k=limit)

            if not docs:
                return f"'{product_query}'에 대한 가격 정보를 찾을 수 없습니다."

            # 검색 결과 포맷팅
            result = f"'{product_query}' 관련 가격 정보:\n\n"

            for idx, doc in enumerate(docs, 1):
                result += f"[{idx}] {doc.metadata.get('product', '상품명 없음')}\n"
                result += f"{doc.page_content}\n"
                result += "-" * 50 + "\n\n"

            return result
        except Exception as e:
            logger.error(f"식료품 가격 검색 오류: {e}")
            return f"가격 정보를 검색하는 중 오류가 발생했습니다: {str(e)}"

    def get_grocery_recommendations(self, product_query: str) -> Dict[str, Any]:
        """식료품 추천 및 가격 비교 정보 반환"""
        try:
            # RAG에서 관련 문서 검색
            docs = self.rag_service.retriever.invoke(product_query, k=3)

            if not docs:
                return {
                    "success": False,
                    "message": f"'{product_query}'에 대한 정보를 찾을 수 없습니다.",
                    "data": None
                }

            # 검색된 문서들의 정보 추출
            recommendations = []
            for doc in docs:
                recommendations.append({
                    "product": doc.metadata.get('product', ''),
                    "category": doc.metadata.get('category', ''),
                    "unit": doc.metadata.get('unit', ''),
                    "min_price": doc.metadata.get('min_price', 0),
                    "max_price": doc.metadata.get('max_price', 0),
                    "details": doc.page_content
                })

            return {
                "success": True,
                "message": f"{len(recommendations)}개의 관련 상품을 찾았습니다.",
                "data": recommendations
            }
        except Exception as e:
            logger.error(f"식료품 추천 오류: {e}")
            return {
                "success": False,
                "message": f"추천 정보를 가져오는 중 오류가 발생했습니다: {str(e)}",
                "data": None
            }
