package vectorizer

import (
	"encoding/json"
	"math"
	"os"
	"time"

	"gopher-fraud-detection/internal/dto"
)

const scale = 10000

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

func clamp01I16(x float32) int16 {
	if x < 0 {
		x = 0
	} else if x > 1 {
		x = 1
	}
	return int16(math.Round(float64(x) * scale))
}

// Vectorize converts a fraud request into a 16-lane int16 vector (dims 14,15 = 0).
// Sentinel: dims 5,6 = -10000 when last_transaction is null.
func (v *Vectorizer) Vectorize(req dto.FraudRequest) [16]int16 {
	var vec [16]int16
	n := v.Norm

	vec[0] = clamp01I16(float32(req.Transaction.Amount) / n.MaxAmount)
	vec[1] = clamp01I16(float32(req.Transaction.Installments) / n.MaxInstallments)
	if req.Customer.AvgAmount > 0 {
		vec[2] = clamp01I16((float32(req.Transaction.Amount) / float32(req.Customer.AvgAmount)) / n.AmountVsAvgRatio)
	}

	t, _ := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	t = t.UTC()
	vec[3] = clamp01I16(float32(t.Hour()) / 23.0)
	wd := int(t.Weekday())
	vec[4] = clamp01I16(float32((wd+6)%7) / 6.0)

	if req.LastTx == nil {
		vec[5] = -scale
		vec[6] = -scale
	} else {
		lastT, _ := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		minutes := float32(t.Sub(lastT).Minutes())
		vec[5] = clamp01I16(minutes / n.MaxMinutes)
		vec[6] = clamp01I16(float32(req.LastTx.KmFromCurrent) / n.MaxKm)
	}

	vec[7] = clamp01I16(float32(req.Terminal.KmFromHome) / n.MaxKm)
	vec[8] = clamp01I16(float32(req.Customer.TxCount24h) / n.MaxTxCount24h)

	if req.Terminal.IsOnline {
		vec[9] = scale
	}
	if req.Terminal.CardPresent {
		vec[10] = scale
	}

	knownMerchant := false
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			knownMerchant = true
			break
		}
	}
	if !knownMerchant {
		vec[11] = scale
	}

	if risk, ok := v.MccRisk[req.Merchant.MCC]; ok {
		vec[12] = clamp01I16(risk)
	} else {
		vec[12] = scale / 2 // default 0.5
	}

	vec[13] = clamp01I16(float32(req.Merchant.AvgAmount) / n.MaxMerchantAvgAmount)

	return vec
}

// TagFromRequest computes the 4-bit partition tag directly from request fields.
//
//	bit0 = has_last_tx
//	bit1 = unknown_merchant
//	bit2 = is_online
//	bit3 = card_present
func TagFromRequest(req dto.FraudRequest) int {
	tag := 0
	if req.LastTx != nil {
		tag |= 1
	}
	knownMerchant := false
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			knownMerchant = true
			break
		}
	}
	if !knownMerchant {
		tag |= 2
	}
	if req.Terminal.IsOnline {
		tag |= 4
	}
	if req.Terminal.CardPresent {
		tag |= 8
	}
	return tag
}
