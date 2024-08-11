.PHONY: all
all: build

.PHONY: run
run: build down up

.PHONY: clean
clean:
	docker compose -f docker-compose.yml down
	docker image prune -af

.PHONY: fclean
fclean: clean
	docker volume prune -af

.PHONY: re
re: fclean all

.PHONY: build
build:
	docker compose -f docker-compose.yml build

.PHONY: up
up: 
	docker compose -f docker-compose.yml up -d

.PHONY: down
down:
	docker compose -f docker-compose.yml down

.PHONY: exec
exec:
	docker compose exec -it openvoice bash

.PHONY: show
show:
	docker ps -a
	@echo
	docker images
	@echo
	docker volume ls
