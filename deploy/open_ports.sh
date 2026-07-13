#!/usr/bin/env bash
# ==============================================================================
# Oracle's Ubuntu images ship with a restrictive iptables INPUT chain that drops
# 80/443 even after you add ingress rules in the cloud console. This opens them
# at the OS level and persists the rule. Run once on the server.
#
#   bash open_ports.sh
# ==============================================================================
set -euo pipefail

echo "==> Opening TCP 80 and 443 in the OS firewall"
# Insert BEFORE the catch-all REJECT rule that Oracle's image ends the chain with.
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT

echo "==> Persisting rules across reboots"
sudo apt-get install -y iptables-persistent netfilter-persistent
sudo netfilter-persistent save

echo "==> Current INPUT chain:"
sudo iptables -L INPUT -n --line-numbers | head -20
echo
echo "Ports 80/443 open. (Also confirm the ingress rules exist in the Oracle"
echo "console: VCN > Security Lists > Default > add 0.0.0.0/0 TCP 80 and 443.)"
