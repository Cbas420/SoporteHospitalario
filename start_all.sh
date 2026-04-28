#!/bin/bash
set -e

echo "=========================================="
echo "  HOSPITAL AI SYSTEM - STARTUP SCRIPT"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker first."
    exit 1
fi

# ============================================
# 1. Start base infrastructure
# ============================================
log_info "1/6 Starting base infrastructure (db, minio, clinical-db)..."
docker-compose up -d db minio clinical-db

# Wait for postgres to be healthy
log_info "Waiting for PostgreSQL to be ready..."
timeout=60
elapsed=0
while ! docker exec hospital-db pg_isready -U hospital -d hospital > /dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $elapsed -ge $timeout ]; then
        log_error "Timeout waiting for PostgreSQL"
        exit 1
    fi
done

# Wait for clinical-db to be healthy
log_info "Waiting for Clinical DB to be ready..."
elapsed=0
while ! docker exec hospital-clinical-db pg_isready -U clinical -d clinical_db > /dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $elapsed -ge $timeout ]; then
        log_error "Timeout waiting for Clinical DB"
        exit 1
    fi
done

log_info "Base infrastructure is ready!"

# ============================================
# 2. Initialize MinIO buckets
# ============================================
log_info "2/6 Initializing MinIO buckets..."
docker-compose up minio-init || log_warn "MinIO init failed, may already be initialized"

# ============================================
# 3. Generate sample data (if not exists)
# ============================================
log_info "3/6 Checking sample data..."

if [ ! -d "data/raw/COVID" ] && [ ! -d "data/raw/Normal" ] && [ ! -d "data/raw/Viral Pneumonia" ]; then
    log_info "Generating sample data for radiology..."
    docker-compose --profile setup run ai-sample-data
else
    log_info "Sample data already exists, skipping..."
fi

# Generate clinical sample data (if not exists)
if [ ! -f "clinical_ai/data/clinical_dataset.csv" ]; then
    log_info "Generating sample data for clinical..."
    docker-compose run clinical-api python clinical_ai/generate_clinical_data.py --num-patients 1000
else
    log_info "Clinical sample data already exists, skipping..."
fi

# ============================================
# 4. Train radiology model (if not exists)
# ============================================
log_info "4/6 Checking radiology model..."

if [ ! -f "models/chest_xray_classifier.keras" ]; then
    log_warn "Radiology model not found. Training now (this may take a while)..."
    docker-compose --profile training run ai-train
else
    log_info "Radiology model already exists, skipping training..."
fi

# ============================================
# 5. Train clinical model (if not exists)
# ============================================
log_info "5/6 Checking clinical model..."

if [ ! -f "clinical_ai/models/clinical_risk_classifier.keras" ]; then
    log_warn "Clinical model not found. Training now..."
    docker-compose run clinical-api python clinical_ai/train_clinical_model.py --epochs 20
else
    log_info "Clinical model already exists, skipping training..."
fi

# ============================================
# 6. Start all services
# ============================================
log_info "6/6 Starting all services (api, clinical-api, flask-app, dashboard, automation, watchdog)..."

docker-compose up -d api clinical-api flask-app dashboard automation watchdog

echo ""
echo "=========================================="
echo "  HOSPITAL AI SYSTEM IS READY!"
echo "=========================================="
echo ""
echo "Available services:"
echo "  Flask App:      http://localhost:5000"
echo "  Radiology API:  http://localhost:8000"
echo "  Clinical API:   http://localhost:8001"
echo "  Dashboard:       http://localhost:8501"
echo "  MinIO Console:  http://localhost:9001"
echo ""
echo "To view logs: docker-compose logs -f [service_name]"
echo "To stop:     docker-compose down"
echo "=========================================="
