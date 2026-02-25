# Medical Appointment Booking Agent

AI-powered medical appointment booking system with natural language chat interface.

## Features

- 🤖 **AI Chat Agent**: Natural language conversation for booking appointments
- 🏥 **Doctor Availability**: Real-time doctor schedules from Aarogya HMIS API
- 📅 **Smart Slot Selection**: Automatic doctor and time slot selection from user input
- 🎨 **Streamlit UI**: Modern web interface for easy interaction
- 🔄 **Session Management**: Persistent chat sessions
- 📱 **Multi-language Support**: Gujarati, Hindi, English

## Tech Stack

- **Backend**: FastAPI + Python
- **AI**: Ollama + Llama 3.2
- **Frontend**: Streamlit
- **MCP**: Model Context Protocol for tool calling
- **External API**: Aarogya HMIS API integration

## Installation

1. **Clone & Setup**:
   ```bash
   git clone <repo>
   cd medical-appointment-agent
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create `.env` file:
   ```env
   ENVIRONMENT=dev
   OLLAMA_BASE_URL=http://localhost:11434
   LLM_MODEL=llama3.2:latest
   EXTERNAL_API_KEY_DEV=your_api_key_here
   ```

3. **Start Ollama** (if not already running):
   ```bash
   # Check if running
   curl http://localhost:11434/api/tags
   
   # If not running, start it
   ollama serve
   ```
   
   **Note**: If you get "bind: Only one usage of each socket address" error, Ollama is already running - skip this step.

## Usage

### Option 1: Streamlit UI (Recommended)

1. **Start Backend**:
   ```bash
   python run.py
   ```

2. **Start Frontend** (in new terminal):
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open**: http://localhost:8501

### Option 2: FastAPI Only

1. **Start Server**:
   ```bash
   python run.py
   ```

2. **API Docs**: http://localhost:8000/docs

3. **Chat Endpoint**: POST /chat

## Example Conversation

```
User: I have fever and cough
Assistant: Shows available doctors...

User: Dr. Sharma at 5pm
Assistant: ✅ Doctor 'Dr. Rajesh Sharma' selected with slot 17:00 - 17:15

User: My name is John Doe, age 25, male, mobile 9876543210, address Ahmedabad, pin 380001
Assistant: ✅ Appointment booked! ID: ABC123
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - System status
- `GET /doctors` - List available doctors
- `POST /chat` - Chat with agent
- `GET /session/{id}` - Get session info
- `DELETE /session/{id}` - Reset session

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Streamlit  │────│   FastAPI   │────│   Ollama    │
│     UI      │    │   Backend   │    │   (LLM)     │
└─────────────┘    └─────────────┘    └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ Aarogya API │
                 │  (External) │
                 └─────────────┘
```

## Development

- **Backend**: `app/` directory
- **Agent Logic**: `app/agent/agent.py`
- **API Client**: `app/api/`
- **UI**: `streamlit_app.py`

## License

MIT License