# Docker Deployment Guide

This guide explains how to use the official Docker images published to GitHub Container Registry.

## Quick Start with Pre-built Image

### Pull the Latest Image

```bash
docker pull ghcr.io/hydropix/translatebookswithllms:latest
```

### Run the Container

```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/translated_files:/app/translated_files \
  -v $(pwd)/logs:/app/logs \
  -e API_ENDPOINT=http://host.docker.internal:11434/api/generate \
  -e DEFAULT_MODEL=qwen3:14b \
  ghcr.io/hydropix/translatebookswithllms:latest
```

Access the web interface at: `http://localhost:5000`

## Available Image Tags

- `latest` - Latest stable version from main branch
- `v1.2.3` - Specific semantic version (e.g., v1.0.0, v2.1.3)
- `v1.2` - Latest patch version of major.minor (e.g., v1.2)
- `v1` - Latest minor and patch of major version
- `main-<sha>` - Specific commit from main branch

## Using Docker Compose

### 1. Create a `.env` file

```bash
cp deployment/.env.docker.example deployment/.env
```

Edit the `.env` file to configure your LLM settings:

```env
API_ENDPOINT=http://host.docker.internal:11434/api/generate
DEFAULT_MODEL=qwen3:14b
LLM_PROVIDER=ollama
PORT=5000
OLLAMA_NUM_CTX=2048
REQUEST_TIMEOUT=900
```

### 2. Create `docker-compose.yml`

```yaml
version: '3.8'

services:
  translate-book:
    image: ghcr.io/hydropix/translatebookswithllms:latest
    ports:
      - "5000:5000"
    volumes:
      - ./translated_files:/app/translated_files
      - ./logs:/app/logs
      - ./data:/app/data
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 3. Start the Service

```bash
docker compose up -d
```

## Using with Remote Ollama Server

If your Ollama server is on a different machine in your local network:

```yaml
version: '3.8'

services:
  translate-book:
    image: ghcr.io/hydropix/translatebookswithllms:latest
    ports:
      - "5000:5000"
    environment:
      - API_ENDPOINT=http://ollama-server.local:11434/api/generate
      - LLM_PROVIDER=ollama
    extra_hosts:
      # Replace with your Ollama server's IP address
      # Example: if your server is at 192.168.1.50
      - "ollama-server.local:YOUR_OLLAMA_SERVER_IP"
    volumes:
      - ./translated_files:/app/translated_files
      - ./logs:/app/logs
```

**Note:** Docker Desktop isolates containers from your local network. Use `extra_hosts` to map hostnames to IP addresses.

## Multi-Platform Support

The images are built for multiple architectures:
- `linux/amd64` - Intel/AMD 64-bit systems
- `linux/arm64` - ARM 64-bit systems (Apple Silicon, Raspberry Pi, etc.)

Docker automatically pulls the correct architecture for your system.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_ENDPOINT` | LLM API endpoint | `http://localhost:11434/api/generate` |
| `DEFAULT_MODEL` | Default LLM model | `qwen3:14b` |
| `LLM_PROVIDER` | Provider (ollama/gemini/openai/openrouter/mistral/deepseek/poe/nim/opencode) | `ollama` |
| `GEMINI_API_KEY` | Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `PORT` | Web server port | `5000` |
| `OLLAMA_NUM_CTX` | Context window size | `2048` |
| `REQUEST_TIMEOUT` | API timeout (seconds) | `900` |
| `MAX_TOKENS_PER_CHUNK` | Tokens per translation chunk | `400` |
| `SIGNATURE_ENABLED` | Add signature to translations | `true` |
| `ATTRIBUTION_PAGE_ENABLED` | Append an attribution page to translated EPUBs | `true` |

## Volume Mounts

### Recommended Volumes

```bash
-v /path/to/translated_files:/app/translated_files  # Output files
-v /path/to/logs:/app/logs                         # Application logs
-v /path/to/data:/app/data                         # Uploads and temp data
```

### Permissions

The container runs as the default user. Ensure mounted directories have appropriate permissions:

```bash
mkdir -p translated_files logs data
chmod 755 translated_files logs data
```

## Using with Ollama

### Scenario 1: Ollama on Host Machine

```bash
docker run -d \
  -p 5000:5000 \
  -e API_ENDPOINT=http://host.docker.internal:11434/api/generate \
  -e DEFAULT_MODEL=qwen3:14b \
  ghcr.io/hydropix/translatebookswithllms:latest
```

**Note**: `host.docker.internal` allows the container to access services on the host.

### Scenario 2: Ollama in Separate Container

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  translate-book:
    image: ghcr.io/hydropix/translatebookswithllms:latest
    ports:
      - "5000:5000"
    environment:
      - API_ENDPOINT=http://ollama:11434/api/generate
      - DEFAULT_MODEL=qwen3:14b
    depends_on:
      - ollama

volumes:
  ollama_data:
```

## Using Cloud LLM Providers

### Gemini

```bash
docker run -d \
  -p 5000:5000 \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_API_KEY=your_api_key_here \
  -e DEFAULT_MODEL=gemini-2.0-flash \
  ghcr.io/hydropix/translatebookswithllms:latest
```

### OpenAI

```bash
docker run -d \
  -p 5000:5000 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=your_api_key_here \
  -e API_ENDPOINT=https://api.openai.com/v1/chat/completions \
  -e DEFAULT_MODEL=gpt-4o \
  ghcr.io/hydropix/translatebookswithllms:latest
```

## Health Check

The container includes a built-in health check:

```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{"status": "ok"}
```

## Viewing Logs

```bash
# Docker logs
docker logs <container_id>

# Application logs (if volume mounted)
tail -f logs/translation.log
```

## Troubleshooting

### Container won't start

1. Check logs: `docker logs <container_id>`
2. Verify port 5000 is available: `netstat -an | grep 5000`
3. Check environment variables are set correctly

### Can't connect to Ollama

1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Use `host.docker.internal` instead of `localhost` in `API_ENDPOINT`
3. Ensure firewall allows container to access host

### Permission denied on volumes

```bash
# Fix permissions
sudo chown -R $(id -u):$(id -g) translated_files logs data
```

## Building Custom Images

If you need to build a custom image:

```bash
# Clone the repository
git clone https://github.com/hydropix/TranslateBookWithLLM.git
cd TranslateBookWithLLM

# Build the image
docker build -f deployment/Dockerfile -t my-custom-translator .

# Run your custom image
docker run -d -p 5000:5000 my-custom-translator
```

## Security Considerations

- **API Keys**: Never commit API keys to `.env` files in version control
- **Network**: Use Docker networks to isolate containers
- **Volumes**: Mount volumes with minimal necessary permissions
- **Updates**: Regularly update to the latest image version for security patches

## GitHub Container Registry

Images are automatically built and published to GitHub Container Registry when:
- New commits are pushed to `main` branch (tagged as `latest`)
- New version tags are created (e.g., `v1.0.0`)

### Pulling Specific Versions

```bash
# Latest version
docker pull ghcr.io/hydropix/translatebookswithllms:latest

# Specific version
docker pull ghcr.io/hydropix/translatebookswithllms:v1.2.3

# Specific commit
docker pull ghcr.io/hydropix/translatebookswithllms:main-abc1234
```

## CI/CD Integration

The project uses GitHub Actions to automatically build and publish Docker images:
- **Workflow**: `.github/workflows/docker-publish.yml`
- **Multi-platform**: Builds for amd64 and arm64
- **Caching**: Uses GitHub Actions cache for faster builds
- **Tagging**: Automatic semantic versioning and latest tags

## Support

For issues related to Docker deployment:
1. Check this documentation
2. Review container logs
3. Open an issue at: https://github.com/hydropix/TranslateBookWithLLM/issues
