# 🐳 Docker & MariaDB 설정 가이드

Enhanced Internal DB RAG 시스템을 위한 Docker 기반 MariaDB 설정 가이드입니다.

## 🚀 빠른 시작

### 1단계: Docker 설치 (최초 1회만)

```bash
# Docker 설치 스크립트 실행
sudo ./install_docker.sh

# 설치 후 사용자 그룹 적용 (터미널 재시작 대신)
newgrp docker

# 설치 확인
docker --version
docker compose version
```

### 2단계: MariaDB 시작

```bash
# MariaDB 컨테이너 시작
./setup_database.sh start

# 상태 확인
./setup_database.sh status
```

### 3단계: API 서버와 연동 테스트

```bash
# API 서버 실행 (다른 터미널에서)
cd /home/ubuntu_euphoria/Desktop/AWS_LOCAL_LLM
python src/main.py

# API 테스트 (또 다른 터미널에서)
python test_internal_db_api.py
```

## 📋 주요 명령어

### 데이터베이스 관리

```bash
./setup_database.sh start      # MariaDB 시작
./setup_database.sh stop       # MariaDB 중지
./setup_database.sh restart    # MariaDB 재시작
./setup_database.sh status     # 상태 확인
./setup_database.sh logs       # 로그 보기
./setup_database.sh reset      # 데이터 초기화
./setup_database.sh dev        # MariaDB + phpMyAdmin 시작
```

### Docker 관리

```bash
docker compose ps              # 컨테이너 상태
docker compose logs mariadb    # MariaDB 로그
docker compose down            # 모든 컨테이너 중지
docker volume ls               # 데이터 볼륨 확인
```

## 🔗 연결 정보

| 항목 | 값 |
|------|-----|
| **Host** | `127.0.0.1` (localhost) |
| **Port** | `53301` |
| **Database** | `sql_db` |
| **Username** | `manager` |
| **Password** | `SqlDba-1` |
| **Charset** | `utf8mb4` |

### 직접 MySQL 클라이언트 연결

```bash
# MySQL 클라이언트가 설치된 경우
mysql -h 127.0.0.1 -P 53301 -u manager -pSqlDba-1 sql_db

# Docker를 통한 연결
docker exec -it rag_mariadb mysql -u manager -pSqlDba-1 sql_db
```

## 🌐 웹 인터페이스 (개발용)

phpMyAdmin이 포함된 개발 모드:

```bash
./setup_database.sh dev
```

- **phpMyAdmin**: http://localhost:8080
- 사용자: `manager`
- 비밀번호: `SqlDba-1`

## 📊 초기 데이터

데이터베이스 시작 시 자동으로 다음 데이터가 설정됩니다:

### `knowledge` 테이블
- 어텐션 메커니즘
- Self-Attention
- FAISS
- Transformer
- RAG
- 벡터 데이터베이스
- 임베딩

### `products` 테이블
- QA 시스템 Pro
- 벡터 검색 엔진
- 문서 임베딩 도구

## 🔧 문제 해결

### Docker 권한 문제
```bash
sudo service docker start
sudo usermod -aG docker $USER
newgrp docker
```

### 포트 충돌
```bash
# 포트 53301 사용 중인 프로세스 확인
sudo lsof -i :53301

# 또는 다른 포트 사용 (docker-compose.yml 수정)
ports:
  - "53302:3306"  # 예: 53302로 변경
```

### 컨테이너가 시작되지 않는 경우
```bash
# 로그 확인
./setup_database.sh logs

# 완전 초기화
./setup_database.sh reset
```

### WSL2 메모리 부족
WSL2의 메모리 제한을 늘리려면 Windows에서 `%USERPROFILE%\\.wslconfig` 파일을 생성:

```ini
[wsl2]
memory=4GB
processors=2
```

## 🚀 성능 최적화

### MariaDB 설정 조정

`docker-compose.yml`의 command 섹션을 수정:

```yaml
command: >
  --character-set-server=utf8mb4
  --collation-server=utf8mb4_unicode_ci
  --innodb-buffer-pool-size=512M        # 메모리에 맞게 조정
  --max-connections=100                 # 연결 수 조정
  --innodb-flush-log-at-trx-commit=2   # 성능 개선
```

### 볼륨 백업

```bash
# 데이터 백업
docker exec rag_mariadb mysqldump -u manager -pSqlDba-1 sql_db > backup.sql

# 데이터 복원
docker exec -i rag_mariadb mysql -u manager -pSqlDba-1 sql_db < backup.sql
```

## 📈 모니터링

### 리소스 사용량 확인
```bash
docker stats rag_mariadb
```

### 연결 수 확인
```bash
docker exec rag_mariadb mysql -u manager -pSqlDba-1 -e "SHOW STATUS LIKE 'Threads_connected';"
```

## 🏗️ 고급 설정

### 환경 변수로 설정 변경

`.env` 파일을 생성하여 기본값 변경:

```bash
# .env 파일
DB_HOST=127.0.0.1
DB_PORT=53301
DB_USER=manager
DB_PASS=SqlDba-1
DB_NAME=sql_db

MYSQL_ROOT_PASSWORD=rootpass123
```

### SSL 연결 활성화

보안이 중요한 환경에서는 SSL 설정을 추가할 수 있습니다.

---

## ✅ 설정 완료 체크리스트

- [ ] Docker 설치 완료 (`docker --version` 확인)
- [ ] MariaDB 컨테이너 시작 (`./setup_database.sh start`)
- [ ] 데이터베이스 연결 확인 (`./setup_database.sh status`)
- [ ] API 서버 연동 테스트 (`python test_internal_db_api.py`)
- [ ] 테이블 데이터 확인 (phpMyAdmin 또는 MySQL 클라이언트)

설정이 완료되면 Enhanced Internal DB RAG API를 사용할 수 있습니다! 🎉