# Docker Deployment Guide

This guide explains how to deploy TranslateBookWithLLM using Docker.

## Prerequisites

- Docker installed (version 20.10 or higher)
- Docker Compose installed (version 1.29 or higher)
- For Ollama: Ollama running on your host machine with models installed

## Quick Start

### 1. Create Environment File

Copy the example environment file and configure it:

```bash
cp .env.docker.example .env
```

Edit `.env` with your settings (see Configuration section below).

### 2. Build and Run

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Check container health
docker-compose ps
```

### 3. Access the Application

Open your browser to: `http://localhost:5000`

## Configuration

### Using Ollama (Local LLM)

**Windows/Mac:**

Edit your `.env` file:

```env
LLM_PROVIDER=ollama
API_ENDPOINT=http://host.docker.internal:11434/api/generate
DEFAULT_MODEL=qwen3:14b
```

**Linux:**

Option 1: Use host IP address

```env
LLM_PROVIDER=ollama
API_ENDPOINT=http://192.168.1.100:11434/api/generate  # Replace with your host IP
DEFAULT_MODEL=qwen3:14b
```

Option 2: Uncomment `extra_hosts` in [docker-compose.yml](docker-compose.yml:64-66):

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Then use:

```env
API_ENDPOINT=http://host.docker.internal:11434/api/generate
```

### Using Gemini (Cloud LLM)

Edit your `.env` file:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

### Using OpenAI (Cloud LLM)

Edit your `.env` file:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
```

## Advanced Configuration

### Custom Port

To run on a different port (e.g., 8080):

```env
PORT=8080
```

Then update the port mapping in your browser: `http://localhost:8080`

### Persistent Data

The Docker setup automatically creates persistent volumes:

- `./translated_files` - Translated output files
- `./logs` - Application logs (if logging is configured)
- `./data` - **Required**: Checkpoint database and job history (see Checkpoint System below)

These folders are created automatically and persist between container restarts.

### Checkpoint and Resume System

TranslateBookWithLLM includes a checkpoint system that allows you to resume interrupted translations. This requires the `data` volume to be properly configured.

**How it works:**

1. **Automatic Checkpointing**: Progress is saved automatically during translation
2. **Job Database**: SQLite database at `data/jobs.db` tracks all translation jobs
3. **Resume Capability**: Use the `/api/resume/<translation_id>` endpoint to continue interrupted translations
4. **File Preservation**: Original uploaded files are kept in `data/uploads/` for resumption

**Volume Configuration:**

The `data` volume is **required** for checkpoint functionality. It's already configured in `docker-compose.yml`:

```yaml
volumes:
  - ./data:/app/data  # Required: for checkpoint/resume functionality and job history
```

**Important Notes:**

- **Data Persistence**: The `data` directory must persist between container restarts
- **Job Retention**: All job history and checkpoints are stored in `data/jobs.db`
- **Backup Recommendation**: Regularly backup the `data` directory (see Data Backup section)
- **Disk Space**: Monitor `data/uploads/` as it stores original files until jobs complete

**Using Resume Feature:**

Via Web Interface:
- Interrupted translations show a "Resume" button
- Click to continue from last checkpoint

Via API:
```bash
curl -X POST http://localhost:5000/api/resume/<translation_id>
```

**Checking Job Status:**

```bash
# List all jobs
curl http://localhost:5000/api/jobs

# Get specific job details
curl http://localhost:5000/api/status/<translation_id>
```

### Resource Limits

To add memory/CPU limits, edit [docker-compose.yml](docker-compose.yml):

```yaml
services:
  translatebook:
    # ... other settings ...
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          memory: 2G
```

## Common Commands

### Start/Stop

```bash
# Start container
docker-compose up -d

# Stop container
docker-compose down

# Restart container
docker-compose restart
```

### View Logs

```bash
# Follow logs in real-time
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100

# View logs for specific service
docker-compose logs translatebook
```

### Rebuild After Code Changes

```bash
# Rebuild and restart
docker-compose up -d --build

# Force complete rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Access Container Shell

```bash
# Open bash shell inside container
docker-compose exec translatebook bash

# Run Python commands
docker-compose exec translatebook python translate.py --help
```

## Using CLI Mode in Docker

While the Docker container runs the web interface by default, you can also use the CLI for batch processing and automation.

### Basic CLI Usage

**View help:**
```bash
docker-compose exec translatebook python translate.py --help
```

**Translate a text file:**
```bash
# First, copy your file into the container's volume
cp my_book.txt ./translated_files/

# Then translate it
docker-compose exec translatebook python translate.py \
  -i /app/translated_files/my_book.txt \
  -o /app/translated_files/my_book_fr.txt \
  -sl English -tl French
```

### EPUB Translation via CLI

```bash
docker-compose exec translatebook python translate.py \
  -i /app/translated_files/input.epub \
  -o /app/translated_files/output_fr.epub \
  -sl English -tl French \
  -m qwen3:14b
```

### SRT Subtitle Translation via CLI

```bash
docker-compose exec translatebook python translate.py \
  -i /app/translated_files/movie.srt \
  -o /app/translated_files/movie_es.srt \
  -sl English -tl Spanish
```

### Using Different LLM Providers via CLI

**Gemini:**
```bash
docker-compose exec translatebook python translate.py \
  -i /app/translated_files/book.txt \
  -o /app/translated_files/book_fr.txt \
  --provider gemini \
  --gemini_api_key $GEMINI_API_KEY \
  -m gemini-3.6-flash
```

**OpenAI:**
```bash
docker-compose exec translatebook python translate.py \
  -i /app/translated_files/book.txt \
  -o /app/translated_files/book_fr.txt \
  --provider openai \
  --openai_api_key $OPENAI_API_KEY \
  --api_endpoint https://api.openai.com/v1/chat/completions \
  -m gpt-4o
```

### Batch Processing with Docker

Create a script for batch translation:

```bash
#!/bin/bash
# batch_translate.sh

FILES=(
  "chapter1.txt"
  "chapter2.txt"
  "chapter3.txt"
)

for file in "${FILES[@]}"; do
  echo "Translating $file..."
  docker-compose exec -T translatebook python translate.py \
    -i "/app/translated_files/$file" \
    -o "/app/translated_files/${file%.txt}_fr.txt" \
    -sl English -tl French
done
```

**Run the batch script:**
```bash
chmod +x batch_translate.sh
./batch_translate.sh
```

### Mounting Custom Directories

To work with files outside the default `translated_files` directory, add a custom volume:

**Edit docker-compose.yml:**
```yaml
volumes:
  - ./translated_files:/app/translated_files
  - ./logs:/app/logs
  - ./data:/app/data
  - /path/to/your/books:/app/books  # Add custom mount
```

**Then use:**
```bash
docker-compose up -d
docker-compose exec translatebook python translate.py \
  -i /app/books/input.epub \
  -o /app/books/output_fr.epub \
  -sl English -tl French
```

## Health Monitoring

The container includes a health check that verifies the API is responding:

```bash
# Check health status
docker-compose ps

# View detailed health status
docker inspect --format='{{json .State.Health}}' translatebook-llm | python -m json.tool
```

Health check endpoint: `http://localhost:5000/api/health`

## Troubleshooting

### Container Won't Start

**Check logs:**

```bash
docker-compose logs
```

**Common issues:**

1. **Port already in use:**
   ```
   Error: bind: address already in use
   ```
   Solution: Change `PORT` in `.env` or stop the conflicting service

2. **Environment variables not loaded:**
   - Ensure `.env` file exists in same directory as `docker-compose.yml`
   - Check file permissions: `chmod 644 .env`

### Cannot Connect to Ollama

**Symptoms:**

```
Connection refused to localhost:11434
```

**Solutions:**

1. **Windows/Mac:** Verify `API_ENDPOINT` uses `host.docker.internal`:
   ```env
   API_ENDPOINT=http://host.docker.internal:11434/api/generate
   ```

2. **Linux:** Use host IP or enable `extra_hosts` in docker-compose.yml

3. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

4. **Check Ollama binding:**
   Ensure Ollama listens on all interfaces, not just localhost.

   Set environment variable:
   ```bash
   export OLLAMA_HOST=0.0.0.0:11434
   ```

### Translation Fails

**Check model availability:**

```bash
# List available models in Ollama
curl http://localhost:11434/api/tags

# Pull a model if needed
docker-compose exec translatebook curl -X POST http://host.docker.internal:11434/api/pull \
  -d '{"name": "qwen3:14b"}'
```

**Verify configuration:**

```bash
# Check environment variables inside container
docker-compose exec translatebook env | grep -E "LLM_PROVIDER|API_ENDPOINT|DEFAULT_MODEL"
```

### Health Check Failing

**View health check logs:**

```bash
docker inspect translatebook-llm | grep -A 10 Health
```

**Test health endpoint manually:**

```bash
docker-compose exec translatebook curl http://localhost:5000/api/health
```

### File Permissions (Linux)

If you encounter permission errors with translated files:

```bash
# Set correct ownership
sudo chown -R $USER:$USER ./translated_files

# Or run with user mapping
docker-compose exec -u $(id -u):$(id -g) translatebook bash
```

## Production Deployment

### Using Environment Variables Directly

Instead of `.env` file, pass variables directly:

```bash
docker-compose up -d \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_API_KEY=your_key \
  -e PORT=8080
```

### Behind a Reverse Proxy

Example nginx configuration:

```nginx
server {
    listen 80;
    server_name translate.example.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker Swarm / Kubernetes

For orchestration, consider:

1. **Using secrets** for API keys
2. **External volumes** for translated files
3. **Load balancing** for multiple replicas
4. **Resource limits** per container

## Security Considerations

1. **Never commit `.env` file** - It contains sensitive API keys
2. **Use secrets management** in production (Docker secrets, Kubernetes secrets)
3. **Limit container permissions** - Run as non-root user if possible
4. **Network isolation** - Use Docker networks to isolate services
5. **Regular updates** - Keep base images and dependencies updated

## Translation Features

### Translation Signatures

All translations include a discrete signature for attribution:

- **EPUB**: Adds Dublin Core metadata (`dc:contributor` with role "trl", `dc:description`) and appends a short attribution page at the end of the book
- **Text Files**: Adds footer with project name and GitHub link
- **SRT Files**: Adds comment at end with attribution

**Configuration:**

Enable/disable via `.env`:

```env
# Enable signatures (default)
SIGNATURE_ENABLED=true

# Disable signatures
SIGNATURE_ENABLED=false

# Keep the OPF metadata attribution but drop the visible EPUB end page
ATTRIBUTION_PAGE_ENABLED=false
```

See [TRANSLATION_SIGNATURE.md](../TRANSLATION_SIGNATURE.md) for details.

## Performance Optimization

### Increase Context Window

For larger chunks and better translations:

```env
OLLAMA_NUM_CTX=16384
MAX_TOKENS_PER_CHUNK=800  # Larger token chunks for better context
```

### Adjust Timeouts

For slower models or large files:

```env
REQUEST_TIMEOUT=1800  # 30 minutes
```

### Enable Auto-adjustment

Let the system optimize chunk sizes:

```env
AUTO_ADJUST_CONTEXT=true
MIN_CHUNK_SIZE=5
MAX_CHUNK_SIZE=100
```

## Data Backup and Migration

### Backing Up Translation Data

It's important to regularly backup your translation data, especially the checkpoint database.

**Backup all persistent data:**
```bash
# Create timestamped backup
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_translatebook_$DATE.tar.gz \
  ./data \
  ./translated_files \
  ./logs

# Or use rsync for incremental backups
rsync -av --delete ./data ./backups/data/
rsync -av --delete ./translated_files ./backups/translated_files/
```

**Backup only checkpoint database:**
```bash
# Stop container first (recommended)
docker-compose stop

# Copy database
cp ./data/jobs.db ./backups/jobs_$(date +%Y%m%d).db

# Restart container
docker-compose start
```

**Automated daily backups:**
```bash
# Add to crontab (crontab -e)
0 2 * * * cd /path/to/project && tar -czf /path/to/backups/backup_$(date +\%Y\%m\%d).tar.gz data translated_files
```

### Restoring from Backup

**Restore all data:**
```bash
# Stop container
docker-compose down

# Extract backup
tar -xzf backup_translatebook_20250114.tar.gz

# Restart container
docker-compose up -d
```

**Restore only database:**
```bash
# Stop container
docker-compose stop

# Restore database
cp ./backups/jobs_20250114.db ./data/jobs.db

# Restart container
docker-compose start
```

### Migrating to a New Server

**On old server:**
```bash
# Create full backup
docker-compose down
tar -czf translatebook_migration.tar.gz \
  data \
  translated_files \
  .env \
  docker-compose.yml

# Transfer file to new server
scp translatebook_migration.tar.gz user@newserver:/path/to/destination/
```

**On new server:**
```bash
# Extract backup
tar -xzf translatebook_migration.tar.gz

# Verify .env configuration (update API endpoints if needed)
nano .env

# Start container
docker-compose up -d

# Verify all jobs are accessible
curl http://localhost:5000/api/jobs
```

### Database Maintenance

**Check database size:**
```bash
docker-compose exec translatebook du -sh /app/data/jobs.db
```

**Clean up completed jobs older than 30 days:**

Currently, there's no automatic cleanup. Monitor disk space and manually manage old jobs:

```bash
# View database content
docker-compose exec translatebook sqlite3 /app/data/jobs.db "SELECT id, status, created_at FROM jobs ORDER BY created_at DESC LIMIT 20;"

# Manually backup and clean if needed
docker-compose stop
cp ./data/jobs.db ./data/jobs_backup.db
# Optionally: manually clean old entries from the database
docker-compose start
```

**Monitor data directory size:**
```bash
# Check total size
du -sh ./data

# Check uploads directory (should clean after job completion)
du -sh ./data/uploads
```

## Cleanup

### Remove Container and Volumes

```bash
# Stop and remove container
docker-compose down

# Remove volumes (WARNING: deletes translated files and checkpoint data)
docker-compose down -v

# Remove images
docker rmi translatebookwithllm_translatebook
```

**Important:** Before removing volumes, ensure you have backups of:
- `./data/jobs.db` - Job history and checkpoints
- `./translated_files` - Completed translations
- `./data/uploads` - Original files for pending jobs

### Clean Up Old Images

```bash
# Remove dangling images
docker image prune

# Remove all unused images
docker image prune -a
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Project README](README.md)
