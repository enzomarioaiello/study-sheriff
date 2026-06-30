# study-sheriff

## Setup instructions
### 1. Clone the repo
On your computer, navigate to the directory you want to put this project via your terminal/cli. Once youre in that directory, copy the command below and run it
```bash
git clone https://github.com/enzomarioaiello/study-sheriff.git
```

## Raspberry Pi dashboard

From the project root on the Raspberry Pi:

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8080 --workers 1
```

Open the dashboard from the Mac at:

```text
http://10.55.0.2:8080
```

Use `--workers 1` only, so the camera and NPU pipeline starts once. If the real
camera, Hailo runtime, or model cannot start, the web server stays alive and the
dashboard shows an offline/error state with no mock occupancy or activity data.
