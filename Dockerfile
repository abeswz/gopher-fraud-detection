FROM golang:1.26.3 AS build

WORKDIR /src

ENV GOEXPERIMENT=simd \
    CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64 \
    GOAMD64=v3

COPY go.mod ./

RUN go mod download

COPY cmd/ ./cmd/

COPY internal/ ./internal/

RUN go build -trimpath -buildvcs=false -ldflags="-s -w" -o /out/fraud-api ./cmd/api

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/fraud-api /fraud-api

COPY index/ /index/

COPY resources/ /resources/

ENV FIRST_TX_INDEX_PATH=/index/first_tx.ivfh \
    SUBSEQ_INDEX_PATH=/index/subsequent_tx.ivfh \
    NORM_PATH=/resources/normalization.json \
    MCC_PATH=/resources/mcc_risk.json

ENTRYPOINT ["/fraud-api"]
