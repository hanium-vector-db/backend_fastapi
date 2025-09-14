#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

from langchain_chroma import Chroma
from models.embedding_handler import EmbeddingHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_vector_db():
    """벡터 DB의 내용을 검사합니다"""
    try:
        # 임베딩 핸들러 초기화
        embedding_handler = EmbeddingHandler()

        # Chroma DB 연결
        db = Chroma(
            persist_directory="./data/vector_db",
            embedding_function=embedding_handler.embeddings
        )

        # 데이터베이스 통계
        collection = db._collection
        count = collection.count()

        print(f"\n=== 벡터 DB 검사 결과 ===")
        print(f"총 문서 수: {count}")

        if count > 0:
            # 일부 문서 샘플 조회
            try:
                results = collection.get(limit=5)

                print(f"\n=== 최근 저장된 문서들 (최대 5개) ===")

                for i, (doc_id, metadata, document) in enumerate(zip(
                    results['ids'],
                    results['metadatas'],
                    results['documents']
                )):
                    print(f"\n--- 문서 {i+1} ---")
                    print(f"ID: {doc_id}")
                    print(f"제목: {metadata.get('title', 'N/A')}")
                    print(f"출처: {metadata.get('source', 'N/A')}")
                    print(f"카테고리: {metadata.get('category', 'N/A')}")
                    print(f"날짜: {metadata.get('date', 'N/A')}")
                    print(f"점수: {metadata.get('score', 'N/A')}")
                    print(f"내용 (앞 200자): {document[:200]}...")
                    print(f"내용 길이: {len(document)} 문자")

            except Exception as e:
                print(f"문서 조회 중 오류: {e}")

        # 테스트 검색
        print(f"\n=== 테스트 검색 ('인공지능') ===")
        retriever = db.as_retriever(search_kwargs={'k': 3})
        docs = retriever.invoke("인공지능")

        print(f"검색 결과: {len(docs)}개")
        for i, doc in enumerate(docs):
            print(f"\n검색결과 {i+1}:")
            print(f"제목: {doc.metadata.get('title', 'N/A')}")
            print(f"출처: {doc.metadata.get('source', 'N/A')}")
            print(f"내용 (앞 150자): {doc.page_content[:150]}...")

    except Exception as e:
        print(f"벡터 DB 검사 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_vector_db()