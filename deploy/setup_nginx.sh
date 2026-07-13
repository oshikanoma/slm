#!/usr/bin/env bash
# ==============================================================================
# Put nginx in front of the Gradio app and (optionally) enable HTTPS.
#
# Usage:
#   bash setup_nginx.sh                       # serve on http://<server-ip>
#   DOMAIN=yourname.duckdns.org EMAIL=you@x.com bash setup_nginx.sh
#                                             # serve on https://<domain> + cert
#
# If DOMAIN is set, the DNS for that domain MUST already point at this server's
# public IP (e.g. set it in the DuckDNS dashboard) before running, or the
# Let's Encrypt challenge will fail.
# ==============================================================================
set -euo pipefail

DOMAIN="${DOMAIN:-}"
EMAIL="${EMAIL:-}"
UPSTREAM="127.0.0.1:7860"

SERVER_NAME="${DOMAIN:-_}"   # nginx uses "_" as the catch-all when no domain

echo "==> Writing nginx site (server_name: ${SERVER_NAME})"
sudo tee /etc/nginx/sites-available/verifier >/dev/null <<CONF
server {
    listen 80;
    server_name ${SERVER_NAME};

    # Gradio needs websockets + larger uploads (documents).
    client_max_body_size 25M;

    location / {
        proxy_pass http://${UPSTREAM};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;   # model verification can take ~60s on CPU
    }
}
CONF

sudo ln -sf /etc/nginx/sites-available/verifier /etc/nginx/sites-enabled/verifier
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
echo "==> nginx up on port 80."

if [ -n "${DOMAIN}" ]; then
  if [ -z "${EMAIL}" ]; then
    echo "!! DOMAIN set but EMAIL not — skipping HTTPS. Re-run with EMAIL=you@x.com to get a cert."
    exit 0
  fi
  echo "==> Installing certbot and requesting a Let's Encrypt cert for ${DOMAIN}"
  sudo apt-get install -y certbot python3-certbot-nginx
  sudo certbot --nginx -d "${DOMAIN}" --non-interactive --agree-tos -m "${EMAIL}" --redirect
  echo "==> HTTPS enabled. Auto-renewal is handled by the certbot systemd timer."
  echo "    Visit: https://${DOMAIN}"
else
  IP="$(curl -s ifconfig.me || echo '<server-ip>')"
  echo "    Visit: http://${IP}"
fi
