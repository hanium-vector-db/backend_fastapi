# 테스트 파일들

이 디렉토리는 AWS Local LLM 프로젝트의 테스트 파일들을 포함합니다.

## 테스트 파일 목록

- `test_auto_rag.py` - 자동 External-Web RAG 기능 테스트
- `test_external_web_rag.py` - External-Web RAG 기본 기능 테스트
- `test_news_search.py` - 뉴스 검색 기능 테스트
- `debug_vector_db.py` - 벡터 DB 디버깅 도구

## 실행 방법

프로젝트 루트 디렉토리에서 실행하세요:

```bash
# 자동 RAG 테스트
python tests/test_auto_rag.py

# External-Web RAG 테스트
python tests/test_external_web_rag.py

# 뉴스 검색 테스트
python tests/test_news_search.py
```

## 주의사항

- 테스트 실행 전에 서버가 실행 중이어야 합니다: `python src/main.py`
- 환경 변수가 올바르게 설정되어 있어야 합니다 (`.env` 파일)