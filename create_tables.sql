-- 뉴스 키워드 테이블
CREATE TABLE IF NOT EXISTS news_keywords (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  keyword VARCHAR(100) NOT NULL,
  is_excluded BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(userid),
  UNIQUE KEY unique_user_keyword (user_id, keyword)
);

-- 영양 목표 테이블
CREATE TABLE IF NOT EXISTS nutrition_goals (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  goal_type VARCHAR(50) NOT NULL,
  target_value DECIMAL(10,2),
  current_value DECIMAL(10,2),
  unit VARCHAR(20),
  target_date DATE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(userid)
);

-- 식단 계획 테이블
CREATE TABLE IF NOT EXISTS diet_plans (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  meal_type VARCHAR(50) NOT NULL,
  meal_date DATE NOT NULL,
  meal_name VARCHAR(200),
  calories INT,
  protein DECIMAL(10,2),
  carbs DECIMAL(10,2),
  fat DECIMAL(10,2),
  notes TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(userid)
);

-- 알림 테이블
CREATE TABLE IF NOT EXISTS notifications (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  notification_type VARCHAR(50) NOT NULL,
  title VARCHAR(200) NOT NULL,
  message TEXT,
  is_read BOOLEAN DEFAULT FALSE,
  priority VARCHAR(20) DEFAULT 'normal',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(userid)
);

-- 일일 리포트 테이블
CREATE TABLE IF NOT EXISTS daily_reports (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  report_date DATE NOT NULL,
  health_score INT,
  finance_summary TEXT,
  activity_summary TEXT,
  medication_compliance DECIMAL(5,2),
  ai_insights TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(userid),
  UNIQUE KEY unique_user_date (user_id, report_date)
);

-- 캘린더 이벤트 테이블
CREATE TABLE IF NOT EXISTS calendar_events (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  event_type VARCHAR(50) NOT NULL,
  title VARCHAR(200) NOT NULL,
  description TEXT,
  event_date DATE NOT NULL,
  event_time TIME,
  is_completed BOOLEAN DEFAULT FALSE,
  reminder_enabled BOOLEAN DEFAULT TRUE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(userid)
);

-- 가격 추적 항목 테이블
CREATE TABLE IF NOT EXISTS price_items (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  item_name VARCHAR(200) NOT NULL,
  category VARCHAR(100),
  current_price DECIMAL(10,2),
  target_price DECIMAL(10,2),
  url VARCHAR(500),
  last_checked DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(userid)
);
