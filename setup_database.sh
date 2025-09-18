#!/usr/bin/bash

# MariaDB 설정 및 실행 스크립트
# 사용법: ./setup_database.sh [start|stop|restart|status|logs|reset]

set -e

DB_CONTAINER="rag_mariadb"
COMPOSE_FILE="docker-compose.yml"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker가 설치되지 않았습니다."
        echo "먼저 다음 명령으로 Docker를 설치하세요:"
        echo "sudo ./install_docker.sh"
        exit 1
    fi

    if ! docker ps &> /dev/null; then
        print_error "Docker가 실행되지 않거나 권한이 없습니다."
        echo "다음 중 하나를 실행하세요:"
        echo "1. sudo service docker start"
        echo "2. newgrp docker (사용자가 docker 그룹에 있는 경우)"
        exit 1
    fi
}

wait_for_db() {
    print_status "MariaDB 시작 대기 중..."
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if docker exec $DB_CONTAINER mysql -h localhost -u manager -pSqlDba-1 -e "SELECT 1;" &> /dev/null; then
            print_success "MariaDB 연결 성공!"
            return 0
        fi

        attempt=$((attempt + 1))
        printf "."
        sleep 2
    done

    echo ""
    print_error "MariaDB 연결 대기 시간이 초과되었습니다."
    return 1
}

start_db() {
    print_status "MariaDB 컨테이너 시작..."

    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "docker-compose.yml 파일을 찾을 수 없습니다."
        exit 1
    fi

    if [ ! -f "init-db.sql" ]; then
        print_error "init-db.sql 파일을 찾을 수 없습니다."
        exit 1
    fi

    docker compose up -d mariadb

    if wait_for_db; then
        print_success "MariaDB가 성공적으로 시작되었습니다!"
        print_status "연결 정보:"
        echo "  - Host: localhost"
        echo "  - Port: 53301"
        echo "  - Database: sql_db"
        echo "  - Username: manager"
        echo "  - Password: SqlDba-1"
        echo ""
        print_status "테스트 명령:"
        echo "  mysql -h 127.0.0.1 -P 53301 -u manager -pSqlDba-1 sql_db"
    else
        print_error "MariaDB 시작에 실패했습니다."
        echo "로그를 확인하세요: ./setup_database.sh logs"
        exit 1
    fi
}

stop_db() {
    print_status "MariaDB 컨테이너 중지..."
    docker compose down
    print_success "MariaDB가 중지되었습니다."
}

restart_db() {
    print_status "MariaDB 컨테이너 재시작..."
    stop_db
    sleep 2
    start_db
}

status_db() {
    print_status "Docker 컨테이너 상태:"
    docker compose ps

    echo ""
    print_status "MariaDB 연결 테스트:"
    if docker exec $DB_CONTAINER mysql -h localhost -u manager -pSqlDba-1 -e "SELECT 'Connection OK' as status, NOW() as timestamp;" 2>/dev/null; then
        print_success "MariaDB 연결 성공!"
    else
        print_warning "MariaDB에 연결할 수 없습니다."
    fi
}

show_logs() {
    print_status "MariaDB 로그:"
    docker compose logs -f mariadb
}

reset_db() {
    print_warning "이 작업은 모든 데이터를 삭제합니다!"
    read -p "계속하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "데이터베이스 초기화 중..."
        docker compose down
        docker volume rm aws_local_llm_mariadb_data 2>/dev/null || true
        start_db
        print_success "데이터베이스가 초기화되었습니다."
    else
        print_status "초기화가 취소되었습니다."
    fi
}

start_with_phpmyadmin() {
    print_status "MariaDB + phpMyAdmin 시작..."
    docker compose --profile dev up -d

    if wait_for_db; then
        print_success "서비스가 시작되었습니다!"
        echo "  - MariaDB: localhost:53301"
        echo "  - phpMyAdmin: http://localhost:8080"
        echo "    (사용자: manager, 비밀번호: SqlDba-1)"
    fi
}

# 메인 실행 부분
case "$1" in
    start)
        check_docker
        start_db
        ;;
    stop)
        check_docker
        stop_db
        ;;
    restart)
        check_docker
        restart_db
        ;;
    status)
        check_docker
        status_db
        ;;
    logs)
        check_docker
        show_logs
        ;;
    reset)
        check_docker
        reset_db
        ;;
    dev)
        check_docker
        start_with_phpmyadmin
        ;;
    *)
        echo "Enhanced Internal DB RAG - MariaDB 관리 스크립트"
        echo ""
        echo "사용법: $0 {start|stop|restart|status|logs|reset|dev}"
        echo ""
        echo "명령어:"
        echo "  start     - MariaDB 컨테이너 시작"
        echo "  stop      - MariaDB 컨테이너 중지"
        echo "  restart   - MariaDB 컨테이너 재시작"
        echo "  status    - 컨테이너 및 연결 상태 확인"
        echo "  logs      - MariaDB 로그 보기 (Ctrl+C로 종료)"
        echo "  reset     - 모든 데이터 삭제 후 초기화"
        echo "  dev       - MariaDB + phpMyAdmin 시작"
        echo ""
        echo "예시:"
        echo "  ./setup_database.sh start"
        echo "  ./setup_database.sh status"
        echo "  ./setup_database.sh dev"
        exit 1
        ;;
esac