-- Enhanced Internal DB RAG 초기 데이터
-- 사용자: manager, 비밀번호: SqlDba-1, 데이터베이스: sql_db

USE sql_db;

-- Knowledge 테이블 생성
CREATE TABLE IF NOT EXISTS knowledge (
    id INT AUTO_INCREMENT PRIMARY KEY,
    term VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    role TEXT,
    details TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_term (term),
    FULLTEXT(term, description, role, details)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 샘플 데이터 삽입
INSERT INTO knowledge (term, description, role, details) VALUES
('어텐션 메커니즘',
 '어텐션은 입력의 중요한 부분에 가중치를 부여해 정보를 통합하는 기법이다. 시퀀스 처리에서 문맥 의존성을 강화한다.',
 '입력 토큰 간 상호연관성을 계산하며 정보 흐름을 개선한다.',
 'Transformer의 핵심 구성요소로 번역·요약 등에서 성능을 끌어올린다.'),

('Self-Attention',
 'Self-Attention은 동일 시퀀스 내 토큰들이 서로를 참조하여 가중합을 구한다. RNN의 순차 의존성을 줄여 병렬화를 가능케 한다.',
 '장기 의존성 문제를 완화하고 각 토큰의 전역 문맥 파악을 돕는다.',
 '멀티헤드로 다양한 표현 공간에서 주의를 분산해 학습을 안정화한다.'),

('FAISS',
 'FAISS는 대규모 벡터에 대한 빠른 유사도 검색을 제공한다. 근사 최근접 탐색을 지원한다.',
 '대규모 임베딩 인덱싱과 빠른 검색을 제공한다.',
 'Facebook AI Research에서 개발되었고 CPU/GPU 백엔드를 제공한다.'),

('Transformer',
 'Transformer는 어텐션 메커니즘만을 사용한 신경망 아키텍처이다. RNN이나 CNN 없이도 우수한 성능을 보여준다.',
 '자연어 처리의 패러다임을 바꾼 혁신적 모델이다.',
 'BERT, GPT 등 현대 언어모델의 기반이 되었다.'),

('RAG',
 'RAG(Retrieval Augmented Generation)는 외부 지식을 검색하여 생성 모델의 답변 품질을 향상시키는 기법이다.',
 '지식베이스에서 관련 정보를 검색하고 이를 바탕으로 답변을 생성한다.',
 'LLM의 한계인 지식 업데이트와 환각 문제를 완화할 수 있다.'),

('벡터 데이터베이스',
 '벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색할 수 있도록 설계된 데이터베이스이다.',
 '임베딩 벡터 간 유사도 검색을 빠르게 수행한다.',
 'Pinecone, Weaviate, Chroma 등이 대표적인 벡터 DB이다.'),

('임베딩',
 '임베딩은 단어, 문장, 문서 등을 고차원 벡터 공간의 점으로 표현하는 기법이다.',
 '의미적 유사성을 벡터 간 거리로 측정할 수 있게 해준다.',
 'Word2Vec, BERT, OpenAI의 text-embedding 모델 등이 활용된다.');

-- 추가 테이블: 제품 정보 (테스트용)
CREATE TABLE IF NOT EXISTS products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    description TEXT,
    price DECIMAL(10, 2),
    features TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    FULLTEXT(name, description, features)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 제품 샘플 데이터
INSERT INTO products (name, category, description, price, features) VALUES
('QA 시스템 Pro', 'AI Software', 'RAG 기반 질의응답 시스템으로 대규모 문서에서 정확한 답변을 제공합니다.', 299.99, '자동 인덱싱, 실시간 검색, 다국어 지원, API 제공'),
('벡터 검색 엔진', 'Database', '고성능 벡터 유사도 검색을 지원하는 전문 데이터베이스입니다.', 499.99, 'FAISS 통합, 분산 처리, REST API, 실시간 업데이트'),
('문서 임베딩 도구', 'AI Tools', '다양한 형식의 문서를 고품질 벡터로 변환하는 도구입니다.', 199.99, '다중 형식 지원, 배치 처리, 클라우드 연동, 자동 청킹');

-- 사용자 권한 설정 (보안 강화)
GRANT SELECT, INSERT, UPDATE, DELETE ON sql_db.* TO 'manager'@'%';
FLUSH PRIVILEGES;

-- 초기 설정 완료 로그
SELECT 'Enhanced Internal DB RAG 초기 설정 완료' AS status,
       COUNT(*) AS knowledge_records FROM knowledge
UNION ALL
SELECT 'Product records', COUNT(*) FROM products;