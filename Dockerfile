FROM golang:1.26.3 AS builder

WORKDIR /app

ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64 \
    GOAMD64=v3

COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ ./cmd/
COPY internal/ ./internal/

RUN go build \
    -trimpath \
    -buildvcs=false \
    -ldflags="-s -w" \
    -o fraud-api \
    ./cmd/api && \
    go build \
    -trimpath \
    -buildvcs=false \
    -ldflags="-s -w" \
    -o lb \
    ./cmd/lb

FROM gcr.io/distroless/static-debian12

COPY --from=builder /app/fraud-api /fraud-api
COPY --from=builder /app/lb /lb
COPY index/ /app/index/
COPY resources/ /app/resources/

ENV INDEX_DIR=/app/index \
    NORM_PATH=/app/resources/normalization.json \
    MCC_PATH=/app/resources/mcc_risk.json

ENTRYPOINT ["/fraud-api"]
