#!/bin/bash
# EC2에서 최초 SSL 인증서 발급 스크립트
# 사용법: ./init-ssl.sh

set -e

# .env에서 변수 로드
if [ ! -f .env ]; then
  echo "❌ .env 파일이 없습니다. .env.example을 복사해서 설정해주세요."
  exit 1
fi
source .env

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
  echo "❌ .env에 DOMAIN과 EMAIL을 설정해주세요."
  exit 1
fi

echo "🔧 도메인: $DOMAIN"

# nginx/app.conf에 도메인 적용
sed -i "s/YOUR_DOMAIN/$DOMAIN/g" nginx/app.conf

# certbot 디렉토리 생성
mkdir -p certbot/conf certbot/www

# 1단계: HTTP만 허용하는 임시 nginx 설정으로 certbot 인증 진행
echo "📦 nginx 시작 (HTTP)..."
docker compose -f docker-compose.prod.yml up -d nginx

echo "🔐 SSL 인증서 발급 중..."
docker compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  -d "$DOMAIN" \
  --email "$EMAIL" \
  --agree-tos \
  --no-eff-email

# 2단계: 전체 서비스 시작 (HTTPS 포함)
echo "🚀 전체 서비스 시작..."
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "✅ 배포 완료!"
echo "   접속 주소: https://$DOMAIN"
