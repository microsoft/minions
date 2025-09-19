FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /app
 
# Install required runtime dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy application files
COPY src/microbot/environment/local_docker/image_builder/dockerShell.py .
COPY src/microbot/environment/local_docker/image_builder/ShellCommunicator.py .

EXPOSE 8080

CMD ["python", "dockerShell.py"]
