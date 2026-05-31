PORT           := 9999
READY_TIMEOUT  := 300

.PHONY: index bench bench-fast \
	profile-vp-serial profile-ivf-serial \
	profile-vp-parallel profile-ivf-parallel \
	trace-vp trace-ivf \
	profile profile-ivf \
	submission

index:
	uv run --project ml ml/build_index.py --algo ivf

index-vp:
	uv run --project ml ml/build_index.py --algo vptree

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

# Serial profiles — algorithm cost, no goroutine contention
profile-vp-serial:
	go test ./internal/search/ -bench=BenchmarkVPKNN_RealIndex -benchtime=15s \
		-cpuprofile=cpu-vp-serial.pprof -memprofile=mem-vp-serial.pprof
	@echo "go tool pprof -http=:8080 cpu-vp-serial.pprof"

profile-ivf-serial:
	go test ./internal/search/ -bench=BenchmarkKNN_RealIndex -benchtime=15s \
		-cpuprofile=cpu-ivf-serial.pprof -memprofile=mem-ivf-serial.pprof
	@echo "go tool pprof -http=:8080 cpu-ivf-serial.pprof"

# Parallel profiles — GOMAXPROCS goroutines, reveals cache thrashing and contention
profile-vp-parallel:
	go test ./internal/search/ -bench=BenchmarkVPKNN_RealIndex_Parallel -benchtime=15s \
		-cpuprofile=cpu-vp-parallel.pprof -memprofile=mem-vp-parallel.pprof \
		-blockprofile=block-vp.pprof -mutexprofile=mutex-vp.pprof
	@echo "go tool pprof -http=:8080 cpu-vp-parallel.pprof"

profile-ivf-parallel:
	go test ./internal/search/ -bench=BenchmarkKNN_RealIndex_Parallel -benchtime=15s \
		-cpuprofile=cpu-ivf-parallel.pprof -memprofile=mem-ivf-parallel.pprof \
		-blockprofile=block-ivf.pprof -mutexprofile=mutex-ivf.pprof
	@echo "go tool pprof -http=:8080 cpu-ivf-parallel.pprof"

# Execution trace — goroutine scheduling, GC, syscall breakdown
trace-vp:
	go test ./internal/search/ -bench=BenchmarkVPKNN_RealIndex_Parallel -benchtime=5s \
		-trace=trace-vp.out
	@echo "go tool trace trace-vp.out"

trace-ivf:
	go test ./internal/search/ -bench=BenchmarkKNN_RealIndex_Parallel -benchtime=5s \
		-trace=trace-ivf.out
	@echo "go tool trace trace-ivf.out"

# Legacy aliases
profile: profile-vp-serial
profile-ivf: profile-ivf-serial

submission:
	@echo "Use: ./references/tools/submission.sh"
	@exit 1
