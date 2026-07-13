# Deploying The Verifier to Oracle Cloud (free, permanent, 24/7)

This runs the **full live model** on a free Oracle Ampere VM, reachable at a real
HTTPS URL. Oracle's **Always Free** tier never charges the card — it's identity
verification only.

**The split:** *you* do Part A (create the account + VM — a web console, ~15 min,
which can't be automated). Then you hand Claude the server IP + SSH key and Claude
does Part B (all the server setup) for you.

---

## Part A — What you do (create the VM)

### 1. Sign up
- Go to <https://www.oracle.com/cloud/free/> → **Start for free**.
- Use a real email; pick your home region (remember which one — free capacity varies).
- Enter the card for verification. **Always Free resources are never billed.**

### 2. Create the VM instance
- In the console: **☰ Menu → Compute → Instances → Create instance**.
- **Name:** `verifier`.
- **Image and shape → Edit → Change shape → Ampere** → `VM.Standard.A1.Flex`.
  Set **4 OCPUs** and **24 GB memory** (all within Always Free).
- **Image:** keep **Canonical Ubuntu** (22.04 or 24.04).
- **SSH keys:** choose **Generate a key pair for me** → **Download private key**
  (and public key). Save the private key file — you'll give it to Claude.
- Click **Create**. Wait ~1 min until it's **RUNNING**.
- Copy the **Public IP address** shown on the instance page.

> **"Out of capacity" error?** Ampere free capacity is popular. Either retry in a
> few minutes, pick a different Availability Domain in the create dialog, or try
> during off-peak hours. This is the one common snag; it clears with retries.

### 3. Open the network ports (in the console)
- On the instance page, click the **subnet** link → **Security Lists** →
  **Default Security List** → **Add Ingress Rules**.
- Add two rules, both **Source `0.0.0.0/0`**, **IP Protocol TCP**,
  **Destination Port** `80` and `443` (one rule each). Save.

### 4. Hand off to Claude
Give Claude:
- the **public IP**, and
- the **private SSH key file** (paste its contents, or save it somewhere and give the path).

That's it for you. Claude takes it from here.

---

## Part B — What Claude does (automated server setup)

Claude SSHes in as `ubuntu@<your-ip>` with the key and runs, in order:

```bash
# 1. Get the deploy scripts onto the server
sudo apt-get update -y && sudo apt-get install -y git
git clone https://github.com/oshikanoma/slm.git /tmp/slm
cd /tmp/slm/deploy

# 2. Open the OS firewall (Oracle's image blocks 80/443 by default)
bash open_ports.sh

# 3. Install everything + run the app as a 24/7 service (downloads the model)
bash setup_oracle.sh

# 4a. Simple: serve on http://<ip>
bash setup_nginx.sh
# 4b. OR with the free domain + HTTPS you chose:
DOMAIN=<yourname>.duckdns.org EMAIL=<you@example.com> bash setup_nginx.sh
```

### The free domain (DuckDNS)
Before step 4b, you set up a free subdomain (no card):
1. Go to <https://www.duckdns.org>, sign in (GitHub/Google).
2. Pick a subdomain, e.g. `tiffany-verifier` → it becomes `tiffany-verifier.duckdns.org`.
3. Set its **IP** to your Oracle public IP and **Update**.
4. Tell Claude the domain; Claude runs step 4b to get the HTTPS cert.

---

## Operating it later

```bash
sudo systemctl status verifier     # is it running?
sudo systemctl restart verifier    # restart
sudo journalctl -u verifier -f     # live logs
```

**Update the site after code changes** (re-pull + restart):
```bash
cd /opt/verifier && git pull && sudo systemctl restart verifier
```

**Optional — better web retrieval:** get a free key at <https://tavily.com>, then
add `Environment=TAVILY_API_KEY=tvly-...` to `/etc/systemd/system/verifier.service`
and `sudo systemctl daemon-reload && sudo systemctl restart verifier`. Without it,
the app uses the free Wikipedia search backend.
