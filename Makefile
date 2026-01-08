# Unfold Makefile
# Common commands for development and deployment

.PHONY: help dev prod build test clean logs

# Default target
help:
	@echo "Unfold - AI-Assisted Reading Platform"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make dev-backend  - Start backend only"
	@echo "  make dev-frontend - Start frontend only"
	@echo "  make test         - Run all tests"
	@echo "  make test-backend - Run backend tests"
	@echo "  make test-e2e     - Run E2E tests"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production environment"
	@echo "  make prod-build   - Build production images"
	@echo "  make prod-down    - Stop production environment"
	@echo ""
	@echo "Utilities:"
	@echo "  make logs         - View all logs"
	@echo "  make logs-backend - View backend logs"
	@echo "  make clean        - Remove containers and volumes"
	@echo "  make shell-backend - Open backend shell"
	@echo "  make db-shell     - Open PostgreSQL shell"

# ===================
# Development
# ===================

dev:
	docker-compose up -d
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend:  http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Neo4j:    http://localhost:7474"

dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

# ===================
# Production
# ===================

prod:
	docker-compose -f docker-compose.prod.yml up -d
	@echo "Production environment started!"

prod-build:
	docker-compose -f docker-compose.prod.yml build --no-cache

prod-down:
	docker-compose -f docker-compose.prod.yml down

prod-pull:
	docker-compose -f docker-compose.prod.yml pull

prod-restart:
	docker-compose -f docker-compose.prod.yml down
	docker-compose -f docker-compose.prod.yml up -d

# ===================
# Testing
# ===================

test: test-backend test-e2e

test-backend:
	cd backend && pytest -v

test-backend-cov:
	cd backend && pytest --cov=app --cov-report=html -v

test-e2e:
	cd frontend && npm run test:e2e

test-e2e-ui:
	cd frontend && npm run test:e2e:ui

# ===================
# Logs
# ===================

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

logs-postgres:
	docker-compose logs -f postgres

logs-neo4j:
	docker-compose logs -f neo4j

# ===================
# Utilities
# ===================

clean:
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.prod.yml down -v --remove-orphans 2>/dev/null || true
	docker system prune -f

shell-backend:
	docker-compose exec backend /bin/sh

db-shell:
	docker-compose exec postgres psql -U postgres -d unfold

neo4j-shell:
	docker-compose exec neo4j cypher-shell -u neo4j -p password

# ===================
# Database
# ===================

db-migrate:
	cd backend && alembic upgrade head

db-rollback:
	cd backend && alembic downgrade -1

db-reset:
	docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS unfold;"
	docker-compose exec postgres psql -U postgres -c "CREATE DATABASE unfold;"

# ===================
# SSL Certificates
# ===================

ssl-generate:
	@mkdir -p nginx/ssl
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout nginx/ssl/privkey.pem \
		-out nginx/ssl/fullchain.pem \
		-subj "/CN=localhost"
	@echo "Self-signed SSL certificates generated in nginx/ssl/"

ssl-letsencrypt:
	@echo "Use certbot for production SSL:"
	@echo "certbot certonly --webroot -w /var/www/html -d your-domain.com"
