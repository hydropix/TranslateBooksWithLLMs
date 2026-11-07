# Docker Testing Guide

This guide explains how to test the Docker deployment of TranslateBookWithLLM.

## Prerequisites

- Docker Desktop installed and running
- For Windows: WSL 2 enabled
- At least 2GB of free disk space

**Note:** This guide uses `docker-compose` (standalone) commands. On newer systems (GitHub Actions, recent Ubuntu), use `docker compose` (with space) instead.

## Quick Start

### Automated Testing (Recommended)

#### Windows

```batch
cd deployment
test_docker.bat
```

#### Linux/macOS

```bash
cd deployment
chmod +x test_docker.sh
./test_docker.sh
```

The script will automatically:
1. ✅ Verify Docker installation
2. ✅ Build the Docker image
3. ✅ Start the container
4. ✅ Wait for health check
5. ✅ Test API endpoints
6. ✅ Display results

---

## Manual Testing

### 1. Build the Image

```bash
cd deployment
docker-compose build
```

Expected output: `deployment-translatebook Built`

### 2. Start the Container

```bash
docker-compose up -d
```

Expected output: `Container translatebook-llm Started`

### 3. Check Container Status

```bash
docker-compose ps
```

Expected status: `Up X seconds (healthy)`

### 4. Test Health Endpoint

```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{
  "message": "Translation API is running",
  "status": "ok",
  "supported_formats": ["txt", "epub", "srt"],
  "translate_module": "loaded"
}
```

### 5. Access Web Interface

Open in browser: [http://localhost:5000](http://localhost:5000)

You should see the translation interface.

---

## Common Commands

### View Logs

```bash
docker-compose logs -f
```

Press `Ctrl+C` to exit.

### Restart Container

```bash
docker-compose restart
```

### Stop Container

```bash
docker-compose down
```

### Rebuild and Restart

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

### Shell Access

```bash
docker-compose exec translatebook bash
```

Type `exit` to leave the container shell.

---

## GitHub Actions CI/CD

Every push to `main` or `dev` branch automatically triggers Docker tests on GitHub's servers.

### Viewing Test Results

1. Go to your GitHub repository
2. Click on **Actions** tab
3. Select the latest workflow run
4. View test results and logs

### Manual Trigger

You can manually trigger tests from GitHub:
1. Go to **Actions** tab
2. Select **Docker Build and Test** workflow
3. Click **Run workflow**
4. Choose branch and click **Run workflow**

---

## Troubleshooting

### Container Keeps Restarting

**Check logs:**
```bash
docker-compose logs
```

**Common causes:**
- Missing `.env` file → Copy `.env.docker.example` to `.env`
- Port 5000 already in use → Change `PORT` in `.env`
- Insufficient memory → Allocate more RAM in Docker Desktop settings

### Health Check Fails

**Symptoms:**
- Container status shows `(unhealthy)`
- Health endpoint returns error

**Solutions:**
1. Wait 40 seconds after startup (grace period)
2. Check if Flask server started:
   ```bash
   docker-compose logs | grep "LLM TRANSLATION SERVER"
   ```
3. Verify port mapping:
   ```bash
   docker-compose ps
   ```
   Should show: `0.0.0.0:5000->5000/tcp`

### Cannot Access Web Interface

**Check firewall:**
- Windows: Allow Docker Desktop through Windows Defender Firewall
- Verify container is running:
  ```bash
  docker ps
  ```

**Try localhost alternatives:**
- http://127.0.0.1:5000
- http://0.0.0.0:5000 (may not work on Windows)

### Docker Daemon Not Running

**Error:** `cannot connect to Docker daemon`

**Solution:**
1. Start Docker Desktop
2. Wait for icon to turn green (bottom-right on Windows, top-right on macOS)
3. Run `docker ps` to verify

---

## Testing Different Configurations

### Test with Gemini Provider

1. Edit `deployment/.env`:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_actual_api_key
   GEMINI_MODEL=gemini-3.6-flash
   ```

2. Restart container:
   ```bash
   docker-compose restart
   ```

3. Upload a file via web interface to test translation

### Test with OpenAI Provider

1. Edit `deployment/.env`:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_actual_api_key
   API_ENDPOINT=https://api.openai.com/v1/chat/completions
   DEFAULT_MODEL=gpt-4o
   ```

2. Restart container:
   ```bash
   docker-compose restart
   ```

### Test with Local Ollama

1. Ensure Ollama is running on your host machine
2. Edit `deployment/.env`:
   ```env
   LLM_PROVIDER=ollama
   # Windows/macOS:
   API_ENDPOINT=http://host.docker.internal:11434/api/generate

   # Linux:
   # API_ENDPOINT=http://172.17.0.1:11434/api/generate

   DEFAULT_MODEL=qwen3:14b
   ```

3. Verify Ollama is accessible from container:
   ```bash
   docker-compose exec translatebook curl http://host.docker.internal:11434/api/tags
   ```

---

## Performance Testing

### Check Resource Usage

```bash
docker stats translatebook-llm
```

Displays CPU, memory, network, and disk I/O in real-time.

### Test Translation Speed

Use a small test file to benchmark:

```bash
# Create test file
echo "This is a test translation." > test.txt

# Upload via API (requires curl and jq)
curl -X POST http://localhost:5000/api/translate \
  -F "file=@test.txt" \
  -F "source_language=English" \
  -F "target_language=Chinese" \
  -F "llm_provider=gemini" \
  -F "gemini_api_key=YOUR_KEY"
```

---

## CI/CD Integration

The GitHub Actions workflow ([.github/workflows/docker-test.yml](.github/workflows/docker-test.yml)) automatically tests:

✅ Docker image builds successfully
✅ Container starts without errors
✅ Health endpoint responds correctly
✅ Web interface is accessible

**Triggered on:**
- Push to `main` or `dev` branches
- Pull requests to `main` or `dev`
- Manual workflow dispatch

**Test duration:** ~2-3 minutes

---

## Advanced Testing

### Test Volume Persistence

1. Start container and create a translation
2. Stop container: `docker-compose down`
3. Start container again: `docker-compose up -d`
4. Check if data persists:
   ```bash
   docker-compose exec translatebook ls -la /app/data
   ```

### Test Checkpoint/Resume Feature

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md#checkpoint-and-resume-system) for checkpoint testing procedures.

---

## Getting Help

If tests fail or you encounter issues:

1. **Check logs first:**
   ```bash
   docker-compose logs --tail=50
   ```

2. **Verify configuration:**
   ```bash
   docker-compose config
   ```

3. **Rebuild from scratch:**
   ```bash
   docker-compose down -v  # Remove volumes
   docker-compose build --no-cache  # Build without cache
   docker-compose up -d
   ```

4. **Report issues:**
   - Include output from `docker-compose logs`
   - Include your `.env` configuration (remove API keys!)
   - Include Docker version: `docker --version`
   - Include OS version

---

## Next Steps

After successful testing:

- ✅ Deploy to production (see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md#production-deployment))
- ✅ Configure environment variables
- ✅ Set up reverse proxy (nginx/Traefik)
- ✅ Enable HTTPS
- ✅ Set up monitoring and logging
