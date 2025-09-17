import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ShellCommunicator import ShellCommunicator

shell = ShellCommunicator("bash")
shell.start_session()

class Message(BaseModel):
    message: str

app = FastAPI()

@app.post("/")
async def receive_message(message: Message):
    command_output = shell.send_command(message.message)
    return {"status": "success", "output": "\n".join(command_output)}

if __name__ == "__main__":
    # Prefer AGENT_PORT, else default 8080
    port = int(os.getenv("AGENT_PORT")  or 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)
