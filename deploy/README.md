Deployment options

- Local macOS
  - Use the provided script: `bash backend/scripts/run_all.sh`
  - To run only server: `uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000`

- Ubuntu 24.04 EC2 (4 vCPU, 16 GB RAM, no GPU)
  1) Copy `spiritual-quest-clean.zip` (optional) or clone the repo to the server, then run:
     ```bash
     curl -fsSL https://raw.githubusercontent.com/<your-repo>/main/deploy/ubuntu_bootstrap.sh -o ubuntu_bootstrap.sh
     chmod +x ubuntu_bootstrap.sh
     sudo APP_DIR=/opt/spiritual-quest APP_USER=ubuntu OPENROUTER_API_KEY=sk-... ./ubuntu_bootstrap.sh
     ```
  2) Service control:
     ```bash
     sudo systemctl status spiritual-quest.service
     sudo systemctl restart spiritual-quest.service
     journalctl -u spiritual-quest.service -n 200 -f
     ```

- Systemd service
  - Runs uvicorn as a system service with environment in `/etc/spiritual-quest.env`.
  - Change PORT/HOST via the env file then `sudo systemctl restart spiritual-quest`.


