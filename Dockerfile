FROM golang:1.26.3 AS builder

WORKDIR /app

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64 \
    GOAMD64=v3

COPY go.mod ./
RUN go mod download

COPY cmd/ ./cmd/
COPY internal/ ./internal/

RUN go build \
    -trimpath \
    -buildvcs=false \
    -ldflags="-s -w" \
    -o fraud-api \
    ./cmd/api

FROM gcr.io/distroless/static-debian12

COPY --from=builder /app/fraud-api /fraud-api
COPY index/ /app/index/
COPY resources/ /app/resources/

ENV FIRST_TX_INDEX_PATH=/app/index/first_tx.ivfh \
    SUBSEQ_ONLINE_INDEX_PATH=/app/index/subseq_online.ivfh \
    SUBSEQ_PHYS_CP_INDEX_PATH=/app/index/subseq_phys_cp.ivfh \
    SUBSEQ_PHYS_NO_CP_INDEX_PATH=/app/index/subseq_phys_no_cp.ivfh \
    NORM_PATH=/app/resources/normalization.json \
    MCC_PATH=/app/resources/mcc_risk.json \
    GOMAXPROCS=1 \
    GOMEMLIMIT=145MiB

ENTRYPOINT ["/fraud-api"]
