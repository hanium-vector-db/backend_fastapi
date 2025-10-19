-- 샘플 데이터 삽입
-- 사용자: geon0078

-- 뉴스 키워드
INSERT INTO news_keywords (user_id, keyword, is_excluded) VALUES
('geon0078', 'AI', FALSE),
('geon0078', '의료', FALSE),
('geon0078', '재테크', FALSE),
('geon0078', '건강', FALSE),
('geon0078', '코스피', FALSE);

-- 건강 관련 데이터
INSERT INTO disease (user_id, name, diagnosed_date, status) VALUES
('geon0078', '고혈압', '2024-01-15', 'active'),
('geon0078', '당뇨', '2024-02-10', 'active');

INSERT INTO medication (user_id, name, dosage, intake_time, alarm_enabled) VALUES
('geon0078', '혈압약', '10mg', '08:00:00', TRUE),
('geon0078', '당뇨약', '5mg', '19:00:00', TRUE),
('geon0078', '비타민D', '1000IU', '08:00:00', TRUE);

-- 영양 목표
INSERT INTO nutrition_goals (user_id, goal_type, target_value, current_value, unit, target_date) VALUES
('geon0078', '일일 칼로리', 2000.00, 1800.00, 'kcal', '2025-12-31'),
('geon0078', '단백질', 150.00, 120.00, 'g', '2025-12-31'),
('geon0078', '체중 감량', 75.00, 78.00, 'kg', '2025-12-31');

-- 식단 계획
INSERT INTO diet_plans (user_id, meal_type, meal_date, meal_name, calories, protein, carbs, fat) VALUES
('geon0078', '아침', '2025-10-19', '계란 토스트', 350, 18.5, 35.0, 12.0),
('geon0078', '점심', '2025-10-19', '닭가슴살 샐러드', 420, 45.0, 25.0, 15.0),
('geon0078', '저녁', '2025-10-19', '연어 구이', 480, 38.0, 22.0, 25.0);

-- 재정 항목
INSERT INTO finance_items (user_id, category, content, created_at, updated_at) VALUES
(1, '수입', '월급', NOW(), NOW()),
(1, '지출', '식비', NOW(), NOW()),
(1, '지출', '의료비', NOW(), NOW()),
(1, '저축', '비상금', NOW(), NOW());

-- 가격 추적
INSERT INTO price_items (user_id, item_name, category, current_price, target_price, last_checked) VALUES
('geon0078', '삼성전자 주식', '주식', 75000.00, 80000.00, NOW()),
('geon0078', '애플 맥북 M3', '전자제품', 1890000.00, 1700000.00, NOW());

-- 알림
INSERT INTO notifications (user_id, notification_type, title, message, is_read, priority) VALUES
('geon0078', 'medication', '복약 알림', '혈압약 복용 시간입니다', FALSE, 'high'),
('geon0078', 'health', '건강 체크', '오늘 혈압을 측정하세요', FALSE, 'normal'),
('geon0078', 'finance', '재정 리마인더', '이번 달 지출을 확인하세요', FALSE, 'normal');

-- 캘린더 이벤트
INSERT INTO calendar_events (user_id, event_type, title, description, event_date, event_time, is_completed) VALUES
('geon0078', 'medical', '병원 예약', '정기 건강검진', '2025-10-25', '14:00:00', FALSE),
('geon0078', 'medication', '약국 방문', '처방약 수령', '2025-10-20', '10:00:00', FALSE),
('geon0078', 'finance', '보험료 납부', '건강보험료 납부', '2025-10-30', NULL, FALSE);

-- 일일 리포트
INSERT INTO daily_reports (user_id, report_date, health_score, finance_summary, activity_summary, medication_compliance, ai_insights) VALUES
('geon0078', '2025-10-19', 85, '오늘 지출: 45,000원 (예산 대비 적정)', '걸음 수: 8,500보, 운동 시간: 30분', 100.00,
'오늘 건강 상태가 양호합니다. 복약을 잘 지키고 있으며, 식단도 목표에 맞게 조절되고 있습니다. 내일은 조금 더 많이 걷는 것을 추천합니다.');
