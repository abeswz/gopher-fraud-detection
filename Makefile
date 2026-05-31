PORT           := 9999
READY_TIMEOUT  := 300

.PHONY: index bench bench-fast profile profile-vp submission

index:
	uv run --project ml ml/build_index.py

bench-fast:
	docker compose --compatibility down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
		printf '.'; sleep 1; i=$$((i+1)); \
		[ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
	done; echo " ready"
	k6 run test/test.js
	@jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors)"' test/results.json

bench: index
	docker compose --compatibility down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
		printf '.'; sleep 1; i=$$((i+1)); \
		[ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
	done; echo " ready"
	k6 run test/test.js
	@jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors)"' test/results.json

# CPU profile da implementação atual (index real, serial)
# Uso: make profile       → perfil do índice atual (auto-detecta IVF/VPT1)
# Resultado: cpu.pprof → analisar com: go tool pprof -http=:8080 cpu.pprof
profile:
	go test ./internal/search/ -bench=BenchmarkVPKNN_RealIndex -benchtime=15s -cpuprofile=cpu.pprof -memprofile=mem.pprof
	@echo "Pronto. Analise com: go tool pprof -http=:8080 cpu.pprof"

profile-ivf:
	go test ./internal/search/ -bench=BenchmarkKNN_RealIndex -benchtime=15s -cpuprofile=cpu-ivf.pprof
	@echo "Pronto. Analise com: go tool pprof -http=:8080 cpu-ivf.pprof"

submission:
	@echo "Use: ./references/tools/submission.sh"
	@exit 1
