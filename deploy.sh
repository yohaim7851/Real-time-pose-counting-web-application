#!/bin/bash
# EC2에서 git clone 후 한 번만 실행하는 초기 배포 스크립트
set -e

echo "======================================"
echo " GymAI Rep Counter - EC2 배포 스크립트"
echo "======================================"

# 1. Docker 설치 확인
if ! command -v docker &> /dev/null; then
  echo "📦 Docker 설치 중..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  echo "⚠️  Docker 설치 완료. 그룹 적용을 위해 재접속 후 다시 실행하세요."
  exit 0
fi

# 2. .env 파일 확인
if [ ! -f .env ]; then
  echo "⚙️  .env 파일 생성 중..."
  cp .env.example .env
  echo ""
  echo "❗ .env 파일에 아래 값을 입력하고 다시 실행하세요:"
  echo "   nano .env"
  echo ""
  echo "   OPENAI_API_KEY=sk-..."
  echo "   DOMAIN=your-domain.com"
  echo "   EMAIL=your@email.com"
  exit 0
fi

source .env

# 3. 모델 가중치 확인
if [ ! -f best_weights_PoseRAC.pth ] && [ ! -f new_weights.pth ]; then
  echo ""
  echo "❌ 모델 가중치 파일이 없습니다."
  echo "   로컬 PC에서 아래 명령으로 전송하세요:"
  echo ""
  echo "   scp -i your-key.pem best_weights_PoseRAC.pth ubuntu@<EC2-IP>:~/gymapp/"
  echo "   scp -i your-key.pem new_weights.pth         ubuntu@<EC2-IP>:~/gymapp/"
  echo "   scp -i your-key.pem all_action.csv          ubuntu@<EC2-IP>:~/gymapp/"
  echo ""
  exit 1
fi

# 4. SSL 인증서 발급 여부 확인
CERT_PATH="certbot/conf/live/$DOMAIN/fullchain.pem"

if [ ! -f "$CERT_PATH" ]; then
  echo "🔐 SSL 인증서 발급 중 (도메인: $DOMAIN)..."
  mkdir -p certbot/conf certbot/www

  # nginx/app.conf에 도메인 적용
  sed -i "s/YOUR_DOMAIN/$DOMAIN/g" nginx/app.conf

  # HTTP nginx 먼저 띄워서 certbot 인증
  docker compose -f docker-compose.prod.yml up -d nginx

  sleep 3

  docker compose -f docker-compose.prod.yml run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    -d "$DOMAIN" \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email

  echo "✅ SSL 인증서 발급 완료"
else
  echo "✅ SSL 인증서 이미 존재"
fi

# 5. 전체 서비스 빌드 & 시작
echo "🚀 서비스 빌드 및 시작 중..."
docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "======================================"
echo "✅ 배포 완료!"
echo "   접속 주소: https://$DOMAIN"
echo ""
echo "유용한 명령어:"
echo "  로그 확인:   docker compose -f docker-compose.prod.yml logs -f app"
echo "  서비스 중지: docker compose -f docker-compose.prod.yml down"
echo "  코드 업데이트 후 재배포: git pull && docker compose -f docker-compose.prod.yml up -d --build app"
echo "======================================"
