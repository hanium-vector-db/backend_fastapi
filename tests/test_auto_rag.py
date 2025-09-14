#!/usr/bin/env python3
import requests
import json
import time

def test_auto_rag():
    """자동 External-Web RAG 기능을 테스트합니다"""

    base_url = "http://localhost:8000/api/v1"

    print("🚀 자동 External-Web RAG 기능 테스트 시작")
    print("="*60)

    # 테스트 시나리오들
    test_scenarios = [
        {
            "query": "삼성전자 AI 반도체 최신 동향",
            "description": "삼성전자 AI 반도체 관련 최신 뉴스 자동 검색 및 분석"
        },
        {
            "query": "인공지능 투자 동향 2024",
            "description": "2024년 AI 투자 트렌드 자동 분석"
        },
        {
            "query": "SK하이닉스 HBM 메모리",
            "description": "SK하이닉스 HBM 메모리 기술 최신 소식"
        }
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 테스트 {i}: {scenario['description']}")
        print("-" * 50)
        print(f"🔍 질의: {scenario['query']}")

        # 자동 RAG 요청
        auto_rag_data = {
            "query": scenario['query'],
            "max_results": 15
        }

        try:
            print("⏰ 자동 웹 검색 및 RAG 처리 시작...")
            start_time = time.time()

            response = requests.post(
                f"{base_url}/external-web/auto-rag",
                json=auto_rag_data,
                timeout=180  # 3분 타임아웃
            )

            end_time = time.time()
            processing_time = end_time - start_time

            if response.status_code == 200:
                result = response.json()

                print(f"✅ 자동 RAG 처리 성공! (처리 시간: {processing_time:.1f}초)")
                print(f"📊 처리 결과:")
                print(f"   • 추가된 청크: {result.get('added_chunks', 0)}개")
                print(f"   • 관련 문서: {len(result.get('relevant_documents', []))}개")
                print(f"   • 상태: {result.get('status', 'unknown')}")
                print(f"   • 검색 쿼리: {result.get('search_query', 'N/A')}")

                # 응답 내용 미리보기
                response_text = result.get('response', '')
                if response_text:
                    print(f"\n📄 생성된 응답 미리보기:")
                    print("-" * 40)
                    # 첫 500자만 표시
                    preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
                    print(preview)
                    print("-" * 40)
                    print(f"📏 전체 응답 길이: {len(response_text)} 문자")

                    # 보고서 형태 체크
                    report_sections = ["📊 핵심 요약", "🔍 상세 분석", "📈 현황", "🎯 주요 포인트"]
                    found_sections = sum(1 for section in report_sections if section in response_text)
                    print(f"📋 보고서 품질: {found_sections}/{len(report_sections)} 섹션 포함")

                    # 관련 문서 정보
                    relevant_docs = result.get('relevant_documents', [])
                    if relevant_docs:
                        print(f"\n📚 참조된 주요 문서:")
                        for j, doc in enumerate(relevant_docs[:3], 1):
                            title = doc.get('title', 'Unknown')[:60] + "..." if len(doc.get('title', '')) > 60 else doc.get('title', 'Unknown')
                            print(f"   {j}. {title}")

                else:
                    print("⚠️  빈 응답이 생성되었습니다.")

            else:
                print(f"❌ 자동 RAG 실패: {response.status_code}")
                print(f"   오류: {response.text}")

        except requests.exceptions.Timeout:
            print("⏰ 요청 시간 초과 (3분)")
        except Exception as e:
            print(f"❌ 요청 오류: {e}")

        print(f"\n✅ 테스트 {i} 완료")
        print("=" * 50)

        if i < len(test_scenarios):
            print("⏳ 다음 테스트를 위해 5초 대기...")
            time.sleep(5)

    print(f"\n🎉 모든 자동 RAG 테스트 완료!")
    print("\n💡 사용법:")
    print(f"   POST {base_url}/external-web/auto-rag")
    print("   Body: {\"query\": \"질문내용\", \"max_results\": 15}")
    print("\n📝 기능:")
    print("   • 질의에 관련된 최신 뉴스를 자동으로 웹에서 검색")
    print("   • 검색된 뉴스를 자동으로 벡터 DB에 저장")
    print("   • 벡터 DB 기반으로 질의에 대한 종합적인 답변 생성")

if __name__ == "__main__":
    test_auto_rag()