import os
import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ShellCommunicator import ShellCommunicator

# Configure logging to see all logs including ShellCommunicator
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Ensure logs go to stdout
    ]
)

# Set specific logger levels
logging.getLogger('ShellCommunicator').setLevel(logging.DEBUG)
logging.getLogger('uvicorn').setLevel(logging.INFO)

shell = ShellCommunicator("bash")
shell.start_session()


class Message(BaseModel):
    message: str


app = FastAPI()


@app.post("/")
async def receive_message(message: Message):
    command_output = shell.send_command(message.message)
    return {"status": "success", "output": command_output}


if __name__ == "__main__":
    # Prefer BOT_PORT, else default 8080
    port = int(os.getenv("BOT_PORT") or 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)
