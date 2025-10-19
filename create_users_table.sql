-- 사용자 테이블 생성
CREATE TABLE IF NOT EXISTS users (
  userid VARCHAR(255) PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  nickname VARCHAR(255),
  password_hash VARCHAR(255) NOT NULL,
  language VARCHAR(10) DEFAULT 'ko',
  consent_health BOOLEAN DEFAULT FALSE,
  consent_finance BOOLEAN DEFAULT FALSE,
  consent_social BOOLEAN DEFAULT FALSE,
  consent_clp BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 회원가입 임시 데이터 테이블 (registration_id로 관리)
CREATE TABLE IF NOT EXISTS temp_registrations (
  registration_id VARCHAR(255) PRIMARY KEY,
  language VARCHAR(10),
  consent_health BOOLEAN,
  consent_finance BOOLEAN,
  consent_social BOOLEAN,
  consent_clp BOOLEAN,
  name VARCHAR(255),
  nickname VARCHAR(255),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  expires_at DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
