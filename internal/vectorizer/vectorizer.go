package vectorizer

import (
	"encoding/json"
	"os"
	"time"

	"gopher-fraud-detection/internal/dto"
)

type Normalization struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKm                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

type Vectorizer struct {
	Norm    Normalization
	MccRisk map[string]float32
}

func Load(normPath, mccPath string) (*Vectorizer, error) {
	normData, err := os.ReadFile(normPath)
	if err != nil {
		return nil, err
	}
	var norm Normalization
	if err := json.Unmarshal(normData, &norm); err != nil {
		return nil, err
	}

	mccData, err := os.ReadFile(mccPath)
	if err != nil {
		return nil, err
	}
	var mccRisk map[string]float32
	if err := json.Unmarshal(mccData, &mccRisk); err != nil {
		return nil, err
	}

	return &Vectorizer{Norm: norm, MccRisk: mccRisk}, nil
}

func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func (v *Vectorizer) Vectorize(req dto.FraudRequest) [14]float32 {
	var vec [14]float32
	n := v.Norm

	vec[0] = clamp(float32(req.Transaction.Amount) / n.MaxAmount)
	vec[1] = clamp(float32(req.Transaction.Installments) / n.MaxInstallments)
	vec[2] = clamp((float32(req.Transaction.Amount) / float32(req.Customer.AvgAmount)) / n.AmountVsAvgRatio)

	t, _ := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	t = t.UTC()
	vec[3] = float32(t.Hour()) / 23.0
	wd := int(t.Weekday())
	vec[4] = float32((wd+6)%7) / 6.0

	if req.LastTx == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		lastT, _ := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		minutes := float32(t.Sub(lastT).Minutes())
		vec[5] = clamp(minutes / n.MaxMinutes)
		vec[6] = clamp(float32(req.LastTx.KmFromCurrent) / n.MaxKm)
	}

	vec[7] = clamp(float32(req.Terminal.KmFromHome) / n.MaxKm)
	vec[8] = clamp(float32(req.Customer.TxCount24h) / n.MaxTxCount24h)

	if req.Terminal.IsOnline {
		vec[9] = 1
	}
	if req.Terminal.CardPresent {
		vec[10] = 1
	}

	known := make(map[string]struct{}, len(req.Customer.KnownMerchants))
	for _, m := range req.Customer.KnownMerchants {
		known[m] = struct{}{}
	}
	if _, ok := known[req.Merchant.ID]; !ok {
		vec[11] = 1
	}

	if risk, ok := v.MccRisk[req.Merchant.MCC]; ok {
		vec[12] = risk
	} else {
		vec[12] = 0.5
	}

	vec[13] = clamp(float32(req.Merchant.AvgAmount) / n.MaxMerchantAvgAmount)

	return vec
}
