#!/usr/bin/env python3
"""
MariaDB 데모 데이터 수정 스크립트
기존 데이터를 정리하고 깔끔하게 다시 삽입합니다.
"""

import pymysql
import sys

# MariaDB 연결 설정
DB_CONFIG = {
    'host': 'localhost',
    'port': 53301,
    'user': 'manager',
    'password': 'SqlDba-1',
    'database': 'sql_db',
    'charset': 'utf8mb4'
}

def get_connection():
    """MariaDB 연결"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        print("✅ MariaDB 연결 성공!")
        return connection
    except Exception as e:
        print(f"❌ MariaDB 연결 실패: {e}")
        return None

def clean_and_insert_data(connection):
    """데이터 정리 후 재삽입"""
    print("\n🧹 기존 데이터 정리 및 재삽입...")

    cursor = connection.cursor()

    try:
        # 외래키 제약 비활성화
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

        # 기존 데이터 삭제
        cursor.execute("DELETE FROM orders")
        cursor.execute("DELETE FROM users")
        cursor.execute("DELETE FROM products")
        cursor.execute("DELETE FROM knowledge")

        # AUTO_INCREMENT 리셋
        cursor.execute("ALTER TABLE orders AUTO_INCREMENT = 1")
        cursor.execute("ALTER TABLE users AUTO_INCREMENT = 1")
        cursor.execute("ALTER TABLE products AUTO_INCREMENT = 1")
        cursor.execute("ALTER TABLE knowledge AUTO_INCREMENT = 1")

        # 1. knowledge 데이터
        knowledge_data = [
            ("어텐션 메커니즘", "어텐션은 입력의 중요한 부분에 가중치를 부여해 정보를 통합하는 기법이다.",
             "입력 토큰 간 상호연관성을 계산하며 정보 흐름을 개선한다.",
             "Transformer의 핵심 구성요소로 번역·요약 등에서 성능을 끌어올린다."),
            ("Self-Attention", "Self-Attention은 동일 시퀀스 내 토큰들이 서로를 참조하여 가중합을 구한다.",
             "장기 의존성 문제를 완화하고 각 토큰의 전역 문맥 파악을 돕는다.",
             "멀티헤드로 다양한 표현 공간에서 주의를 분산해 학습을 안정화한다."),
            ("FAISS", "FAISS는 대규모 벡터에 대한 빠른 유사도 검색을 제공한다.",
             "대규모 임베딩 인덱싱과 빠른 검색을 제공한다.",
             "Facebook AI Research에서 개발되었고 CPU/GPU 백엔드를 제공한다."),
            ("RAG", "Retrieval Augmented Generation은 외부 지식을 활용한 생성 모델이다.",
             "외부 문서 검색과 언어 생성을 결합하여 정확성을 높인다.",
             "질의응답, 요약, 대화 시스템 등 다양한 NLP 태스크에 활용된다."),
            ("Transformer", "인코더-디코더 구조의 어텐션 기반 신경망 아키텍처",
             "순차적 처리 없이 병렬 처리가 가능하여 학습 속도가 빠르다.",
             "BERT, GPT 등 현대 언어 모델의 기반이 되는 핵심 기술이다.")
        ]

        cursor.executemany("""
        INSERT INTO knowledge (term, description, role, details)
        VALUES (%s, %s, %s, %s)
        """, knowledge_data)
        print(f"   ✅ knowledge: {len(knowledge_data)}개 레코드")

        # 2. products 데이터
        products_data = [
            ("QA 시스템 Pro", "AI Software", "RAG 기반 질의응답 시스템으로 대규모 문서에서 정확한 답변을 제공합니다.", 299.99, "자동 인덱싱, 실시간 검색, 다국어 지원, API 제공"),
            ("벡터 검색 엔진", "Database", "고성능 벡터 유사도 검색을 지원하는 전문 데이터베이스입니다.", 499.99, "FAISS 통합, 분산 처리, REST API, 실시간 업데이트"),
            ("문서 임베딩 도구", "AI Tools", "다양한 형식의 문서를 고품질 벡터로 변환하는 도구입니다.", 199.99, "다중 형식 지원, 배치 처리, 클라우드 연동, 자동 청킹"),
            ("AI 챗봇 빌더", "AI Software", "코딩 없이 고성능 AI 챗봇을 구축할 수 있는 플랫폼입니다.", 399.99, "드래그앤드롭 UI, 다중 모델 지원, 웹훅 연동, 분석 대시보드"),
            ("스마트 문서 분석기", "Document Processing", "PDF, Word, Excel 등 다양한 문서를 자동으로 분석하고 요약합니다.", 149.99, "OCR 지원, 자동 분류, 키워드 추출, 요약 생성")
        ]

        cursor.executemany("""
        INSERT INTO products (name, category, description, price, features)
        VALUES (%s, %s, %s, %s, %s)
        """, products_data)
        print(f"   ✅ products: {len(products_data)}개 레코드")

        # 3. users 데이터
        users_data = [
            ("admin", "admin@company.com", "관리자", "2024-01-15 14:30:00"),
            ("analyst", "analyst@company.com", "분석가", "2024-01-15 13:45:00"),
            ("viewer", "viewer@company.com", "조회자", "2024-01-14 16:20:00"),
            ("developer", "dev@company.com", "개발자", "2024-01-15 10:15:00"),
            ("manager", "manager@company.com", "매니저", "2024-01-15 11:30:00")
        ]

        cursor.executemany("""
        INSERT INTO users (username, email, role, last_login)
        VALUES (%s, %s, %s, %s)
        """, users_data)
        print(f"   ✅ users: {len(users_data)}개 레코드")

        # 외래키 제약 활성화
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

        # 4. orders 데이터 (이제 안전하게 삽입 가능)
        orders_data = [
            (2, 1, 1, 299.99, "완료"),   # analyst가 QA 시스템 Pro 주문
            (3, 2, 2, 999.98, "진행중"), # viewer가 벡터 검색 엔진 2개 주문
            (1, 3, 1, 199.99, "대기"),   # admin이 문서 임베딩 도구 주문
            (4, 4, 1, 399.99, "완료"),   # developer가 AI 챗봇 빌더 주문
            (5, 5, 3, 449.97, "배송중"), # manager가 스마트 문서 분석기 3개 주문
            (2, 1, 1, 299.99, "완료"),   # analyst가 QA 시스템 Pro 추가 주문
            (3, 4, 1, 399.99, "대기")    # viewer가 AI 챗봇 빌더 주문
        ]

        cursor.executemany("""
        INSERT INTO orders (user_id, product_id, quantity, total_price, status)
        VALUES (%s, %s, %s, %s, %s)
        """, orders_data)
        print(f"   ✅ orders: {len(orders_data)}개 레코드")

        connection.commit()
        print("\n✅ 모든 데이터 삽입 완료!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        connection.rollback()
        return False

    finally:
        cursor.close()

    return True

def verify_final_data(connection):
    """최종 데이터 확인"""
    print("\n🔍 최종 데이터 확인...")

    cursor = connection.cursor()

    tables = ['knowledge', 'products', 'users', 'orders']

    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   📊 {table}: {count}개 레코드")
        except Exception as e:
            print(f"   ❌ {table} 확인 실패: {e}")

    # JOIN 쿼리 테스트
    try:
        cursor.execute("""
        SELECT o.id, u.username, p.name, o.quantity, o.total_price, o.status
        FROM orders o
        JOIN users u ON o.user_id = u.id
        JOIN products p ON o.product_id = p.id
        LIMIT 3
        """)
        rows = cursor.fetchall()
        print(f"\n   📋 주문 JOIN 테스트:")
        for row in rows:
            print(f"      주문#{row[0]}: {row[1]}이(가) {row[2]} {row[3]}개 주문 - {row[4]}원 ({row[5]})")

    except Exception as e:
        print(f"   ❌ JOIN 테스트 실패: {e}")

    cursor.close()

def main():
    print("🚀 MariaDB 데모 데이터 수정")
    print("=" * 40)

    connection = get_connection()
    if not connection:
        return False

    try:
        # 데이터 정리 및 재삽입
        if clean_and_insert_data(connection):
            # 최종 확인
            verify_final_data(connection)
            print("\n🎉 데모 데이터 수정 완료!")
            return True
        else:
            return False

    finally:
        connection.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)