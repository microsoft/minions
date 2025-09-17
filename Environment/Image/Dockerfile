FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /app
 
# Install required runtime dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy application files
COPY Environment/Image/docker_shell.py .
COPY Environment/Image/ShellCommunicator.py .

EXPOSE 8080

CMD ["python", "docker_shell.py"]
