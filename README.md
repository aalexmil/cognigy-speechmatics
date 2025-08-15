# Cognigy Voice Gateway + Speechmatics Realtime STT Bridge

This project is a small FastAPI + WebSocket service that connects **Cognigy Voice Gateway** to the **Speechmatics** real-time speech-to-text API.

It receives audio from the Voice Gateway, streams it to Speechmatics, and returns live transcripts (interim + final) back to the gateway.

---

## Features

- **Realtime transcription** via Speechmatics WebSocket API  
- Supports interim and final results  
- Endpoint detection & clean session shutdown  
- Detailed logging for troubleshooting  

---

## Requirements

- Python 3.9+  
- A Speechmatics API key (store in `SPEECHMATICS_API_KEY` env var or AWS SSM Parameter Store)  
- (Optional) AWS credentials if fetching keys from Parameter Store  

---

## Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/aalexmil/cognigny-speechmatics.git
   cd cognigy-speechmatics
   export SPEECHMATICS_API_KEY=your_api_key_here
## Run the service
uvicorn main:app --host 0.0.0.0 --port 3001
