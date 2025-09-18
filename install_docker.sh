#!/bin/bash

# Docker 설치 스크립트 for WSL2 Ubuntu
# 실행: chmod +x install_docker.sh && sudo ./install_docker.sh

set -e

echo "🐳 WSL2에서 Docker 설치 시작..."

# 1. 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
apt update && apt upgrade -y

# 2. 필요한 패키지 설치
echo "🔧 의존성 패키지 설치..."
apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    apt-transport-https

# 3. Docker GPG 키 추가
echo "🔑 Docker GPG 키 추가..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# 4. Docker 저장소 추가
echo "📋 Docker 저장소 추가..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. 패키지 리스트 업데이트
apt update

# 6. Docker 설치
echo "🐳 Docker 설치 중..."
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 7. Docker 서비스 시작
echo "🚀 Docker 서비스 시작..."
service docker start

# 8. 현재 사용자를 docker 그룹에 추가
if [ "$SUDO_USER" ]; then
    usermod -aG docker $SUDO_USER
    echo "✅ 사용자 '$SUDO_USER'를 docker 그룹에 추가했습니다."
    echo "⚠️  변경사항을 적용하려면 터미널을 다시 시작하거나 다음 명령을 실행하세요:"
    echo "   newgrp docker"
else
    echo "⚠️  SUDO_USER가 설정되지 않았습니다. 수동으로 docker 그룹에 추가하세요:"
    echo "   sudo usermod -aG docker \$USER"
fi

# 9. Docker 버전 확인
echo "📋 설치된 Docker 버전:"
docker --version
docker compose version

# 10. Docker 상태 확인
echo "🔍 Docker 상태 확인:"
service docker status

echo ""
echo "🎉 Docker 설치 완료!"
echo ""
echo "다음 단계:"
echo "1. 터미널을 다시 시작하거나 'newgrp docker' 실행"
echo "2. 'docker run hello-world'로 설치 확인"
echo "3. MariaDB 컨테이너 실행: 'docker compose up -d'"