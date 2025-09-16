FROM python:3.11-slim

WORKDIR /app

# Install any dependencies
RUN pip install uvicorn
RUN pip install fastapi
RUN pip install pydantic

# Copy application files
COPY Environment/docker_shell.py .
COPY Environment/ShellCommunicator.py .

EXPOSE 8080

CMD ["python", "docker_shell.py"]
