FROM golang:1.26 AS builder

WORKDIR /app

COPY go.mod ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -trimpath \
    -buildvcs=false \
    -ldflags="-s -w" \
    -o fraud-api \
    ./cmd/api

FROM gcr.io/distroless/static-debian12

COPY --from=builder /app/fraud-api /fraud-api
COPY index/ /app/index/
COPY resources/ /app/resources/

ENV INDEX_PATH=/app/index/references.bin
ENV NORM_PATH=/app/resources/normalization.json
ENV MCC_PATH=/app/resources/mcc_risk.json
ENV GOMAXPROCS=2

ENTRYPOINT ["/fraud-api"]
