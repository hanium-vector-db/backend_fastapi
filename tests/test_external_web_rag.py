#!/usr/bin/env python3
import requests
import json
import time

def test_external_web_rag():
    """External-Web RAG 기능을 종합적으로 테스트합니다"""

    base_url = "http://localhost:8000/api/v1"

    print("🧪 External-Web RAG 기능 테스트 시작")
    print("="*60)

    # 테스트 시나리오들
    test_scenarios = [
        {
            "topic": "인공지능",
            "queries": [
                "인공지능의 최신 동향에 대해 설명해주세요",
                "한국의 AI 기술 발전 현황은 어떻습니까?",
                "AI 관련 투자와 정책 동향을 알려주세요"
            ]
        },
        {
            "topic": "삼성전자 AI",
            "queries": [
                "삼성전자의 AI 기술 개발 현황을 분석해주세요",
                "삼성전자 AI 반도체 사업 전략은 무엇인가요?"
            ]
        }
    ]

    for scenario in test_scenarios:
        print(f"\n📋 테스트 시나리오: {scenario['topic']}")
        print("-" * 40)

        # 1. 주제 업로드 테스트
        print(f"1️⃣ 주제 업로드 테스트: '{scenario['topic']}'")

        upload_data = {
            "topic": scenario['topic'],
            "max_results": 15
        }

        try:
            response = requests.post(f"{base_url}/external-web/upload-topic", json=upload_data, timeout=60)

            if response.status_code == 200:
                upload_result = response.json()
                print(f"✅ 업로드 성공!")
                print(f"   추가된 청크: {upload_result.get('added_chunks', 0)}개")
                print(f"   메시지: {upload_result.get('message', 'N/A')}")
            else:
                print(f"❌ 업로드 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                continue

        except Exception as e:
            print(f"❌ 업로드 요청 오류: {e}")
            continue

        # 잠시 대기 (서버 처리 시간 확보)
        time.sleep(2)

        # 2. RAG 질의응답 테스트
        print(f"\n2️⃣ RAG 질의응답 테스트")

        for i, query in enumerate(scenario['queries'], 1):
            print(f"\n🔍 질문 {i}: {query}")

            query_data = {
                "prompt": query,
                "top_k": 8
            }

            try:
                response = requests.post(f"{base_url}/external-web/rag-query", json=query_data, timeout=120)

                if response.status_code == 200:
                    query_result = response.json()

                    print(f"✅ 질의응답 성공!")
                    print(f"📊 응답 길이: {len(query_result.get('response', ''))} 문자")
                    print(f"🔗 관련 문서: {len(query_result.get('relevant_documents', []))}개")

                    # 응답 내용 미리보기
                    response_text = query_result.get('response', '')
                    if response_text:
                        print(f"\n📄 응답 미리보기 (앞 300자):")
                        print("-" * 30)
                        print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
                        print("-" * 30)

                        # 보고서 형태 체크
                        report_sections = ["📊 핵심 요약", "🔍 상세 분석", "📈 현황 및 동향", "🎯 주요 포인트"]
                        found_sections = sum(1 for section in report_sections if section in response_text)
                        print(f"📋 보고서 형태 점수: {found_sections}/{len(report_sections)} 섹션 포함")
                    else:
                        print("⚠️  빈 응답 반환됨")

                else:
                    print(f"❌ 질의응답 실패: {response.status_code}")
                    print(f"   오류: {response.text}")

            except Exception as e:
                print(f"❌ 질의응답 요청 오류: {e}")

            time.sleep(1)  # 각 질의 간 대기

        print(f"\n✅ '{scenario['topic']}' 시나리오 테스트 완료")
        print("=" * 40)
        time.sleep(3)  # 시나리오 간 대기

    print(f"\n🎉 전체 테스트 완료!")

if __name__ == "__main__":
    test_external_web_rag()