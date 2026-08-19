#!/bin/bash
# ============================================================
# Sets up GitHub auto-deploy on your Hetzner server.
# Run this ONCE on your server: bash setup_github_deploy.sh
# ============================================================
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[OK]${NC}  $1"; }
warn(){ echo -e "${YELLOW}[!!]${NC}  $1"; }

echo ""
echo "============================================================"
echo "  GitHub Auto-Deploy Setup"
echo "============================================================"
echo ""

# Generate a dedicated SSH key for GitHub Actions
KEY_FILE="$HOME/.ssh/github_deploy"

if [ -f "$KEY_FILE" ]; then
    warn "Deploy key already exists at $KEY_FILE"
else
    ok "Generating SSH key pair for GitHub Actions..."
    ssh-keygen -t ed25519 -f "$KEY_FILE" -N "" -C "github-actions-deploy"
    ok "Key generated"
fi

# Add the public key to authorized_keys so GitHub can SSH in
if ! grep -qf "$KEY_FILE.pub" "$HOME/.ssh/authorized_keys" 2>/dev/null; then
    cat "$KEY_FILE.pub" >> "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    ok "Public key added to authorized_keys"
else
    ok "Public key already in authorized_keys"
fi

# Make sure the repo is a git repo with a remote
cd /root/zoqira-ai
if ! git remote -v | grep -q origin; then
    warn "No git remote found. Set one up first:"
    warn "  git remote add origin https://github.com/YOURNAME/zoqira-ai.git"
    exit 1
fi

REPO_URL=$(git remote get-url origin)
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
SERVER_USER=$(whoami)

echo ""
echo "============================================================"
echo "  Add these 3 secrets to GitHub"
echo "============================================================"
echo ""
echo "  Go to: GitHub repo → Settings → Secrets and variables"
echo "          → Actions → New repository secret"
echo ""
echo "  ┌─────────────────┬──────────────────────────────────────┐"
echo "  │ Secret name     │ Value                                │"
echo "  ├─────────────────┼──────────────────────────────────────┤"
printf "  │ HETZNER_HOST    │ %-36s │\n" "$SERVER_IP"
printf "  │ HETZNER_USER    │ %-36s │\n" "$SERVER_USER"
echo "  │ HETZNER_SSH_KEY │ (see below)                          │"
echo "  └─────────────────┴──────────────────────────────────────┘"
echo ""
echo "  HETZNER_SSH_KEY — copy everything including the dashes:"
echo ""
cat "$KEY_FILE"
echo ""
echo "============================================================"
echo "  After adding the secrets, push any change to trigger"
echo "  the first deploy:"
echo ""
echo "    git add . && git commit -m 'setup auto-deploy' && git push"
echo ""
echo "  Then watch: GitHub repo → Actions tab"
echo "============================================================"
echo ""
