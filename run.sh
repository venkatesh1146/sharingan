#!/bin/bash
# Market Pulse - Development Startup Script
# Runs all services needed for local development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables from .env.local (only simple key=value pairs)
# Skip complex values like JSON arrays - let pydantic-settings handle those
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$key" ]] && continue
    # Skip lines with JSON arrays or complex values
    [[ "$value" =~ ^\[.*\]$ ]] && continue
    # Export simple values
    export "$key=$value" 2>/dev/null || true
done < .env.local

# Default values
API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8000}
REDIS_URL=${REDIS_URL:-redis://localhost:6379}
MONGODB_URL=${MONGODB_URL:-mongodb://localhost:27017}
CELERY_BROKER_URL=${CELERY_BROKER_URL:-redis://localhost:6379/0}
CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND:-redis://localhost:6379/1}

# PID file directory
PID_DIR=".pids"
mkdir -p $PID_DIR

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if a service is running
check_service() {
    local service=$1
    local pid_file="$PID_DIR/$service.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            return 0  # Running
        fi
    fi
    return 1  # Not running
}

# Activate virtual environment
activate_venv() {
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Please create one with: python -m venv .venv"
        exit 1
    fi
}

# Start Redis (using Docker)
start_redis() {
    print_status "Starting Redis..."
    if docker ps --format '{{.Names}}' | grep -q '^market-pulse-redis$'; then
        print_warning "Redis is already running"
    else
        docker run -d --name market-pulse-redis \
            -p 6379:6379 \
            -v redis_data:/data \
            redis:7-alpine \
            redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru \
            > /dev/null 2>&1 || docker start market-pulse-redis > /dev/null 2>&1
        print_success "Redis started on port 6379"
    fi
}

# Start MongoDB (using Docker)
start_mongodb() {
    print_status "Starting MongoDB..."
    if docker ps --format '{{.Names}}' | grep -q '^market-pulse-mongodb$'; then
        print_warning "MongoDB is already running"
    else
        docker run -d --name market-pulse-mongodb \
            -p 27017:27017 \
            -v mongodb_data:/data/db \
            -e MONGO_INITDB_DATABASE=market_intelligence \
            mongo:7.0 \
            > /dev/null 2>&1 || docker start market-pulse-mongodb > /dev/null 2>&1
        print_success "MongoDB started on port 27017"
    fi
    # Wait for MongoDB to be ready
    sleep 2
}

# Start Celery Worker (background, logs to terminal and file)
start_celery_worker() {
    print_status "Starting Celery Worker..."
    if check_service "celery-worker"; then
        print_warning "Celery Worker is already running"
    else
        # Run celery worker with logs to file and stdout
        celery -A app.celery_app.celery_config worker \
            --loglevel=info \
            --concurrency=4 \
            --queues=default,news,snapshots,indices,maintenance \
            --logfile=logs/celery-worker.log \
            2>&1 &
        echo $! > "$PID_DIR/celery-worker.pid"
        print_success "Celery Worker started (PID: $!) - Logs: logs/celery-worker.log"
    fi
}

# Start Celery Beat (background, logs to terminal and file)
start_celery_beat() {
    print_status "Starting Celery Beat..."
    if check_service "celery-beat"; then
        print_warning "Celery Beat is already running"
    else
        celery -A app.celery_app.celery_config beat \
            --loglevel=info \
            --logfile=logs/celery-beat.log \
            2>&1 &
        echo $! > "$PID_DIR/celery-beat.pid"
        print_success "Celery Beat started (PID: $!) - Logs: logs/celery-beat.log"
    fi
}

# Start Celery Worker (foreground, for standalone use)
start_celery_worker_fg() {
    print_status "Starting Celery Worker (foreground)..."
    celery -A app.celery_app.celery_config worker --loglevel=info --concurrency=4
}

# Start Celery Beat (foreground, for standalone use)
start_celery_beat_fg() {
    print_status "Starting Celery Beat (foreground)..."
    celery -A app.celery_app.celery_config beat --loglevel=info
}

# Start Flower (background, logs to terminal and file)
start_flower() {
    print_status "Starting Flower..."
    if check_service "flower"; then
        print_warning "Flower is already running"
    else
        FLOWER_UNAUTHENTICATED_API=true celery -A app.celery_app.celery_config flower \
            --port=5555 \
            --logfile=logs/flower.log \
            2>&1 &
        echo $! > "$PID_DIR/flower.pid"
        print_success "Flower started on http://localhost:5555 (PID: $!) - Logs: logs/flower.log"
    fi
}

# Start Flower (foreground, for standalone use)
start_flower_fg() {
    print_status "Starting Flower (foreground)..."
    FLOWER_UNAUTHENTICATED_API=true celery -A app.celery_app.celery_config flower --port=5555
}

# Start FastAPI server
start_api() {
    print_status "Starting FastAPI server..."
    uvicorn app.main:app --host ${API_HOST} --port ${API_PORT} --reload
}

# Start FastAPI server in debug mode
start_api_debug() {
    print_status "Starting FastAPI server in DEBUG mode..."
    DEBUG=true uvicorn app.main:app \
        --host ${API_HOST} \
        --port ${API_PORT} \
        --reload \
        --log-level debug \
        --access-log
}

# Trigger initial data population
trigger_initial_data() {
    print_status "Triggering initial data population..."
    sleep 3  # Wait for Celery worker to be ready
    
    # Check if we can reach the Celery broker
    if redis-cli -u $REDIS_URL ping > /dev/null 2>&1; then
        print_status "Triggering news fetch..."
        python -c "
from app.celery_app.tasks.news_tasks import fetch_and_process_news
from app.celery_app.tasks.snapshot_tasks import generate_market_snapshot
from app.celery_app.tasks.indices_tasks import fetch_indices_data

# Trigger all tasks
print('Queuing news fetch task...')
fetch_and_process_news.delay(limit=20)

print('Queuing indices fetch task...')
fetch_indices_data.delay()

print('Queuing snapshot generation task...')
generate_market_snapshot.delay(force=True)

print('All initial tasks queued successfully!')
" 2>&1
        print_success "Initial data population tasks queued"
    else
        print_warning "Could not reach Redis, skipping initial data population"
    fi
}

# Tail celery logs
tail_logs() {
    print_status "Tailing Celery logs (Ctrl+C to stop)..."
    tail -f logs/celery-worker.log logs/celery-beat.log 2>/dev/null || echo "No log files found yet"
}

# Stop all services
stop_all() {
    print_status "Stopping all services..."
    
    # Stop Celery Worker
    if [ -f "$PID_DIR/celery-worker.pid" ]; then
        kill $(cat "$PID_DIR/celery-worker.pid") 2>/dev/null || true
        rm -f "$PID_DIR/celery-worker.pid"
        print_success "Celery Worker stopped"
    fi
    
    # Stop Celery Beat
    if [ -f "$PID_DIR/celery-beat.pid" ]; then
        kill $(cat "$PID_DIR/celery-beat.pid") 2>/dev/null || true
        rm -f "$PID_DIR/celery-beat.pid"
        print_success "Celery Beat stopped"
    fi
    
    # Stop Flower
    if [ -f "$PID_DIR/flower.pid" ]; then
        kill $(cat "$PID_DIR/flower.pid") 2>/dev/null || true
        rm -f "$PID_DIR/flower.pid"
        print_success "Flower stopped"
    fi
    
    # Stop Docker containers
    docker stop market-pulse-redis market-pulse-mongodb 2>/dev/null || true
    print_success "Docker containers stopped"
    
    print_success "All services stopped"
}

# Show status of all services
status() {
    echo -e "\n${BLUE}=== Market Pulse Services Status ===${NC}\n"
    
    # Check Redis
    if docker ps --format '{{.Names}}' | grep -q '^market-pulse-redis$'; then
        echo -e "Redis:         ${GREEN}RUNNING${NC}"
    else
        echo -e "Redis:         ${RED}STOPPED${NC}"
    fi
    
    # Check MongoDB
    if docker ps --format '{{.Names}}' | grep -q '^market-pulse-mongodb$'; then
        echo -e "MongoDB:       ${GREEN}RUNNING${NC}"
    else
        echo -e "MongoDB:       ${RED}STOPPED${NC}"
    fi
    
    # Check Celery Worker
    if check_service "celery-worker"; then
        echo -e "Celery Worker: ${GREEN}RUNNING${NC} (PID: $(cat $PID_DIR/celery-worker.pid))"
    else
        echo -e "Celery Worker: ${RED}STOPPED${NC}"
    fi
    
    # Check Celery Beat
    if check_service "celery-beat"; then
        echo -e "Celery Beat:   ${GREEN}RUNNING${NC} (PID: $(cat $PID_DIR/celery-beat.pid))"
    else
        echo -e "Celery Beat:   ${RED}STOPPED${NC}"
    fi
    
    # Check Flower
    if check_service "flower"; then
        echo -e "Flower:        ${GREEN}RUNNING${NC} (PID: $(cat $PID_DIR/flower.pid)) - http://localhost:5555"
    else
        echo -e "Flower:        ${RED}STOPPED${NC}"
    fi
    
    echo ""
}

# Print usage
usage() {
    echo -e "\n${BLUE}Market Pulse - Development Startup Script${NC}"
    echo -e "\nUsage: ./run.sh [command]\n"
    echo "Commands:"
    echo "  all         Start all services (Redis, MongoDB, Celery, API) with initial data"
    echo "  debug       Start all services with API in debug mode (verbose logging)"
    echo "  api         Start only the FastAPI server"
    echo "  celery      Start Celery Worker and Beat"
    echo "  worker      Start only Celery Worker (foreground)"
    echo "  beat        Start only Celery Beat (foreground)"
    echo "  flower      Start Flower monitoring UI (http://localhost:5555)"
    echo "  infra       Start only infrastructure (Redis, MongoDB)"
    echo "  logs        Tail Celery worker and beat logs"
    echo "  populate    Trigger data population tasks (news, indices, snapshot)"
    echo "  stop        Stop all background services"
    echo "  status      Show status of all services"
    echo "  docker      Run all services using docker-compose"
    echo "  docker-dev  Run all services with dev profile (includes monitoring UIs)"
    echo "  help        Show this help message"
    echo ""
    echo "Celery logs are saved to logs/ directory. Use './run.sh logs' to tail them."
    echo "Press Ctrl+C to gracefully stop services."
    echo ""
}

# Create logs directory
mkdir -p logs

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    print_status "Shutting down services..."
    
    # Kill Celery processes
    if [ -f "$PID_DIR/celery-worker.pid" ]; then
        local pid=$(cat "$PID_DIR/celery-worker.pid")
        kill $pid 2>/dev/null || true
        # Also kill the tee process group
        pkill -P $pid 2>/dev/null || true
        rm -f "$PID_DIR/celery-worker.pid"
    fi
    
    if [ -f "$PID_DIR/celery-beat.pid" ]; then
        local pid=$(cat "$PID_DIR/celery-beat.pid")
        kill $pid 2>/dev/null || true
        pkill -P $pid 2>/dev/null || true
        rm -f "$PID_DIR/celery-beat.pid"
    fi
    
    if [ -f "$PID_DIR/flower.pid" ]; then
        local pid=$(cat "$PID_DIR/flower.pid")
        kill $pid 2>/dev/null || true
        pkill -P $pid 2>/dev/null || true
        rm -f "$PID_DIR/flower.pid"
    fi
    
    # Kill any remaining celery processes started by this script
    pkill -f "celery -A app.celery_app" 2>/dev/null || true
    
    print_success "Cleanup complete"
    exit 0
}

# Main script
case "${1:-all}" in
    all)
        echo -e "\n${BLUE}=== Starting All Market Pulse Services ===${NC}\n"
        activate_venv
        start_redis
        start_mongodb
        
        # Set up cleanup trap
        trap cleanup SIGINT SIGTERM
        
        start_celery_worker
        start_celery_beat
        
        # Give Celery a moment to start
        sleep 3
        
        # Trigger initial data population in background
        trigger_initial_data &
        
        echo ""
        print_success "All background services started!"
        echo -e "${YELLOW}Celery logs: logs/celery-worker.log, logs/celery-beat.log${NC}"
        echo -e "${YELLOW}Use './run.sh logs' in another terminal to tail logs${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
        echo ""
        start_api
        ;;
    debug)
        echo -e "\n${BLUE}=== Starting All Market Pulse Services (DEBUG MODE) ===${NC}\n"
        activate_venv
        start_redis
        start_mongodb
        
        # Set up cleanup trap
        trap cleanup SIGINT SIGTERM
        
        start_celery_worker
        start_celery_beat
        
        # Give Celery a moment to start
        sleep 3
        
        # Trigger initial data population in background
        trigger_initial_data &
        
        echo ""
        print_success "All background services started!"
        echo -e "${YELLOW}Celery logs: logs/celery-worker.log, logs/celery-beat.log${NC}"
        echo -e "${YELLOW}Use './run.sh logs' in another terminal to tail logs${NC}"
        echo -e "${RED}API running in DEBUG mode with verbose logging${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
        echo ""
        start_api_debug
        ;;
    api)
        activate_venv
        start_api
        ;;
    celery)
        echo -e "\n${BLUE}=== Starting Celery Services ===${NC}\n"
        activate_venv
        
        # Set up cleanup trap
        trap cleanup SIGINT SIGTERM
        
        start_celery_worker
        start_celery_beat
        
        echo ""
        print_success "Celery services running. Logs shown above."
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        
        # Keep script running to show logs
        wait
        ;;
    worker)
        activate_venv
        start_celery_worker_fg
        ;;
    beat)
        activate_venv
        start_celery_beat_fg
        ;;
    flower)
        activate_venv
        start_flower_fg
        ;;
    infra)
        start_redis
        start_mongodb
        echo -e "\nInfrastructure services started!"
        ;;
    logs)
        tail_logs
        ;;
    populate)
        echo -e "\n${BLUE}=== Triggering Data Population ===${NC}\n"
        activate_venv
        trigger_initial_data
        ;;
    stop)
        stop_all
        ;;
    status)
        status
        ;;
    docker)
        print_status "Starting all services with docker-compose..."
        docker-compose up --build
        ;;
    docker-dev)
        print_status "Starting all services with docker-compose (dev profile)..."
        docker-compose --profile dev --profile monitoring up --build
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
