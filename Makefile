PORT          := 9999
READY_TIMEOUT := 300

.PHONY: index bench bench-fast bench-stress bench-monitor profile profile-parallel submission

index:
	go run ./cmd/build_index/ resources/references.json.gz index/

bench-fast:
	docker compose --compatibility down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
		printf '.'; sleep 1; i=$$((i+1)); \
		[ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
	done; echo " ready"
	k6 run test/test.js
	@jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors)"' test/results.json

bench-monitor:
	docker compose --compatibility down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
		printf '.'; sleep 1; i=$$((i+1)); \
		[ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
	done; echo " ready"
	bash test/bench-monitor.sh
	@jq -r '"p99:\(.p99_ms)ms p99.9:\(.p999_ms)ms max:\(.max_ms)ms score:\(.final_score) FP:\(.fp) FN:\(.fn) ERR:\(.errs)"' test/stress-results.json

bench-stress:
	docker compose --compatibility down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
		printf '.'; sleep 1; i=$$((i+1)); \
		[ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
	done; echo " ready"
	k6 run test/stress.js
	@jq -r '"p99:\(.p99_ms)ms p99.9:\(.p999_ms)ms max:\(.max_ms)ms score:\(.final_score) FP:\(.fp) FN:\(.fn) ERR:\(.errs)"' test/stress-results.json

bench: index
	docker compose --compatibility down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
		printf '.'; sleep 1; i=$$((i+1)); \
		[ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
	done; echo " ready"
	k6 run test/test.js
	@jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors)"' test/results.json

profile:
	go test ./internal/search/ -bench=BenchmarkKNN_RealIndex -benchtime=15s \
		-cpuprofile=cpu.pprof -memprofile=mem.pprof
	@echo "go tool pprof -http=:8080 cpu.pprof"

profile-parallel:
	go test ./internal/search/ -bench=BenchmarkKNN_RealIndex_Parallel -benchtime=15s \
		-cpuprofile=cpu-parallel.pprof -memprofile=mem-parallel.pprof \
		-blockprofile=block.pprof -mutexprofile=mutex.pprof
	@echo "go tool pprof -http=:8080 cpu-parallel.pprof"

submission:
	@echo "Use: ./references/tools/submission.sh"
	@exit 1
