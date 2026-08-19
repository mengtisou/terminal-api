#!/bin/bash
# ============================================================
# Zoqira-AI — one-shot server setup for Linux (Ubuntu/Debian)
# Run as root: bash setup_server.sh
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[OK]${NC}  $1"; }
warn(){ echo -e "${YELLOW}[!!]${NC}  $1"; }
err(){ echo -e "${RED}[XX]${NC}  $1"; exit 1; }

echo ""
echo "============================================================"
echo "  Zoqira-AI Trading Terminal — Server Setup"
echo "============================================================"
echo ""

# ── 1. detect location ───────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ok "Project directory: $SCRIPT_DIR"

# ── 2. system packages ───────────────────────────────────────
ok "Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip nginx certbot python3-certbot-nginx supervisor curl ufw git 2>&1 | tail -3
ok "System packages ready"

# ── 3. Python deps ───────────────────────────────────────────
ok "Installing Python packages (this takes ~2 minutes)..."
cd "$SCRIPT_DIR"
grep -v -i "MetaTrader5" requirements.txt > /tmp/req_server.txt
pip3 install -r /tmp/req_server.txt --break-system-packages -q
ok "Python packages installed"

# ── 4. .env file ─────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    warn ".env not found — creating template"
    cat > "$SCRIPT_DIR/.env" << 'ENV'
# ── AI providers ──────────────────────────────────────────────
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# ── Data ──────────────────────────────────────────────────────
TWELVEDATA_KEY=
FINNHUB_KEY=
OANDA_TOKEN=
OANDA_BASE=https://api-fxpractice.oanda.com

# ── Server ────────────────────────────────────────────────────
ALLOWED_ORIGINS=*
ALLOW_SYNTHETIC=0
CHEAP_CHAIN=gemini
STATE_DIR=/tmp

# ── Optional protection ───────────────────────────────────────
# ACCESS_KEY=your_secret_password
ENV
    warn "IMPORTANT: Edit $SCRIPT_DIR/.env and add your API keys, then re-run this script"
    warn "  nano $SCRIPT_DIR/.env"
    exit 0
fi
ok ".env found"

# ── 5. test keys ─────────────────────────────────────────────
ok "Testing keys..."
cd "$SCRIPT_DIR"
python3 -c "
import sys; sys.path.insert(0,'.')
from app.config import GEMINI_API_KEY, ANTHROPIC_API_KEY
print('  Gemini  :', 'SET' if GEMINI_API_KEY else 'MISSING')
print('  Claude  :', 'SET' if ANTHROPIC_API_KEY else 'missing (optional)')
" 2>/dev/null || true

# ── 6. supervisor service ────────────────────────────────────
PORT=${PORT:-8000}
ok "Setting up supervisor (port $PORT)..."

cat > /etc/supervisor/conf.d/zoqira.conf << EOF
[program:zoqira]
command=python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
directory=$SCRIPT_DIR
autostart=true
autorestart=true
startretries=5
startsecs=10
stopwaitsecs=30
stderr_logfile=/var/log/zoqira.err.log
stdout_logfile=/var/log/zoqira.out.log
environment=HOME="/root",PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EOF

supervisorctl reread   > /dev/null 2>&1 || true
supervisorctl update   > /dev/null 2>&1 || true
supervisorctl restart zoqira > /dev/null 2>&1 || supervisorctl start zoqira > /dev/null 2>&1 || true
sleep 3

# ── 7. firewall ──────────────────────────────────────────────
ufw allow ssh    > /dev/null 2>&1 || true
ufw allow 80     > /dev/null 2>&1 || true
ufw allow 443    > /dev/null 2>&1 || true
ufw allow $PORT  > /dev/null 2>&1 || true
ufw --force enable > /dev/null 2>&1 || true
ok "Firewall configured"

# ── 8. nginx ─────────────────────────────────────────────────
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo "────────────────────────────────────────────"
echo "  Domain setup (optional but recommended)"
echo "────────────────────────────────────────────"
echo ""
read -p "  Do you have a domain to use (e.g. api.yourdomain.com)? [y/N] " USE_DOMAIN
USE_DOMAIN=${USE_DOMAIN:-n}

if [[ "$USE_DOMAIN" =~ ^[Yy] ]]; then
    read -p "  Enter the domain (e.g. api.yourdomain.com): " DOMAIN
    if [ -n "$DOMAIN" ]; then
        cat > /etc/nginx/sites-available/zoqira << EOF
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 10M;
    location / {
        proxy_pass http://localhost:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
EOF
        ln -sf /etc/nginx/sites-available/zoqira /etc/nginx/sites-enabled/zoqira
        rm -f /etc/nginx/sites-enabled/default
        nginx -t && nginx -s reload
        ok "Nginx configured for $DOMAIN"

        read -p "  Get free HTTPS certificate? [Y/n] " GET_CERT
        GET_CERT=${GET_CERT:-y}
        if [[ "$GET_CERT" =~ ^[Yy] ]]; then
            read -p "  Your email for Let's Encrypt: " EMAIL
            certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" && ok "HTTPS enabled"
        fi
        API_URL="https://$DOMAIN"
    fi
else
    API_URL="http://$SERVER_IP:$PORT"
fi

# ── 9. health check ──────────────────────────────────────────
echo ""
ok "Waiting for service to start..."
sleep 5

if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    ok "Health check passed"
    STATUS="RUNNING"
else
    warn "Health check failed — check logs: tail -50 /var/log/zoqira.err.log"
    STATUS="CHECK LOGS"
fi

# ── 10. summary ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete  ($STATUS)"
echo "============================================================"
echo ""
echo "  API URL :  ${API_URL:-http://$SERVER_IP:$PORT}"
echo "  Health  :  ${API_URL:-http://$SERVER_IP:$PORT}/health"
echo ""
echo "  Connect your frontend:"
echo ""
echo "  https://terminal-api-two.vercel.app/?api=${API_URL:-http://$SERVER_IP:$PORT}"
echo ""
echo "  Useful commands:"
echo "    supervisorctl status zoqira       — check status"
echo "    supervisorctl restart zoqira      — restart"
echo "    tail -f /var/log/zoqira.err.log   — live logs"
echo "    nano $SCRIPT_DIR/.env            — edit keys"
echo ""
echo "  To update:"
echo "    cd $SCRIPT_DIR && git pull && supervisorctl restart zoqira"
echo "============================================================"
echo ""
