#!/bin/bash
# ============================================================
# Zoqira-AI — Full Hetzner setup
# Works alongside an existing nginx/PHP site without touching it
# Run as root: bash setup_hetzner.sh
# ============================================================
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[OK]${NC}  $1"; }
warn(){ echo -e "${YELLOW}[!!]${NC}  $1"; }
err(){ echo -e "${RED}[XX]${NC}  $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8000

echo ""
echo "============================================================"
echo "  Zoqira-AI — Hetzner Setup"
echo "============================================================"
echo ""
ok "Project: $SCRIPT_DIR"
ok "Port: $PORT"

# ── 1. packages ──────────────────────────────────────────────
ok "Installing packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip supervisor curl certbot python3-certbot-nginx git 2>&1 | tail -2
ok "Packages ready"

# ── 2. Python deps ───────────────────────────────────────────
ok "Installing Python packages (~2 min)..."
cd "$SCRIPT_DIR"
grep -v -i "MetaTrader5" requirements.txt > /tmp/req_zoqira.txt
pip3 install -r /tmp/req_zoqira.txt --break-system-packages -q
ok "Python packages installed"

# ── 3. .env ──────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    warn ".env not found — creating template"
    cat > "$SCRIPT_DIR/.env" << 'ENV'
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
TWELVEDATA_KEY=
FINNHUB_KEY=
OANDA_TOKEN=
OANDA_BASE=https://api-fxpractice.oanda.com
ALLOWED_ORIGINS=*
ALLOW_SYNTHETIC=0
CHEAP_CHAIN=gemini
STATE_DIR=/tmp
ENV
    echo ""
    warn "Add your API keys now:"
    warn "  nano $SCRIPT_DIR/.env"
    warn "Then re-run:  bash $SCRIPT_DIR/setup_hetzner.sh"
    echo ""
    exit 0
fi
ok ".env found"

# ── 4. supervisor ────────────────────────────────────────────
ok "Setting up supervisor service..."
cat > /etc/supervisor/conf.d/zoqira.conf << EOF
[program:zoqira]
command=python3 -m uvicorn app.main:app --host 127.0.0.1 --port $PORT --workers 1
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

supervisorctl reread  > /dev/null 2>&1 || true
supervisorctl update  > /dev/null 2>&1 || true
supervisorctl restart zoqira > /dev/null 2>&1 \
    || supervisorctl start zoqira > /dev/null 2>&1 || true
ok "Supervisor configured"

# ── 5. nginx — new site only, realflylink untouched ──────────
echo ""
echo "────────────────────────────────────────────"
echo "  API domain"
echo "────────────────────────────────────────────"
echo "  Your main site: realflylink.com"
echo "  Suggestion    : api.realflylink.com"
echo ""
read -p "  Enter subdomain for the API (e.g. api.realflylink.com): " API_DOMAIN

if [ -z "$API_DOMAIN" ]; then
    SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
    warn "No domain entered — using IP: $SERVER_IP:$PORT"
    warn "Add a subdomain later for HTTPS"
    API_URL="http://$SERVER_IP:$PORT"
    # open the port directly
    ufw allow $PORT > /dev/null 2>&1 || true
else
    # Write ONLY the new site config — realflylink stays untouched
    cat > /etc/nginx/sites-available/zoqira-api << EOF
server {
    listen 80;
    server_name $API_DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
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

    ln -sf /etc/nginx/sites-available/zoqira-api \
           /etc/nginx/sites-enabled/zoqira-api

    nginx -t && nginx -s reload
    ok "Nginx configured for $API_DOMAIN (realflylink.com unchanged)"

    # HTTPS
    read -p "  Get free HTTPS certificate for $API_DOMAIN? [Y/n] " GET_CERT
    GET_CERT=${GET_CERT:-y}
    if [[ "$GET_CERT" =~ ^[Yy] ]]; then
        read -p "  Your email: " EMAIL
        certbot --nginx -d "$API_DOMAIN" \
            --non-interactive --agree-tos -m "$EMAIL" \
            --redirect && ok "HTTPS enabled" || warn "Certbot failed — try manually later"
    fi

    API_URL="https://$API_DOMAIN"
fi

# ── 6. firewall ──────────────────────────────────────────────
ufw allow ssh   > /dev/null 2>&1 || true
ufw allow 80    > /dev/null 2>&1 || true
ufw allow 443   > /dev/null 2>&1 || true
ufw --force enable > /dev/null 2>&1 || true
ok "Firewall configured"

# ── 7. health check ──────────────────────────────────────────
ok "Waiting for service..."
sleep 6
if curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
    ok "Health check passed"
else
    warn "Health check failed — check: tail -30 /var/log/zoqira.err.log"
fi

# ── 8. GitHub deploy key ─────────────────────────────────────
KEY_FILE="$HOME/.ssh/github_deploy_zoqira"
if [ ! -f "$KEY_FILE" ]; then
    ssh-keygen -t ed25519 -f "$KEY_FILE" -N "" -C "github-actions-zoqira" -q
    cat "$KEY_FILE.pub" >> "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
fi
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

# ── 9. summary ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete"
echo "============================================================"
echo ""
echo "  API URL  : $API_URL"
echo "  Health   : $API_URL/health"
echo ""
echo "  Connect frontend:"
echo "  https://terminal-api-two.vercel.app/?api=$API_URL"
echo ""
echo "  ── GitHub Auto-Deploy ──────────────────────────────────"
echo "  Add these secrets to your GitHub repo:"
echo "  (Settings → Secrets → Actions → New repository secret)"
echo ""
printf "  HETZNER_HOST    = %s\n" "$SERVER_IP"
printf "  HETZNER_USER    = %s\n" "root"
echo   "  HETZNER_SSH_KEY = (private key below — copy ALL lines)"
echo ""
cat "$KEY_FILE"
echo ""
echo "  Then push to main and watch GitHub → Actions tab"
echo ""
echo "  ── Useful commands ─────────────────────────────────────"
echo "  supervisorctl status zoqira        — check running"
echo "  supervisorctl restart zoqira       — restart"
echo "  tail -f /var/log/zoqira.err.log    — live logs"
echo "  nano $SCRIPT_DIR/.env             — edit keys"
echo "  nginx -t && nginx -s reload        — reload nginx"
echo "============================================================"
echo ""
