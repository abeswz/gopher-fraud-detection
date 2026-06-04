package service

import (
	"math"
	"time"

	"gopher-fraud-detection/internal/dto"
)

// RawFeatures indices (referenced by generated raw_tree.go rtFeature array):
// 0–8: Amount,AmountRatio,Installments,TxCount24h,KmFromHome,IsKnownMerchant,MccRiskScore,MerchantAvgAmount,MerchantAmountRatio
// 9–20: HourOfDay,IsOnline,CardPresent,LastKmFromCurrent,LastTimeDeltaSec,AmountOverMax,InstallmentsNorm,TxVelocity,IsSafeMCC,IsRiskyMCC,AmountNormalized,CustomerAvgNormalized
type RawFeatures struct {
	Amount                float32
	AmountRatio           float32
	Installments          float32
	TxCount24h            float32
	KmFromHome            float32
	IsKnownMerchant       float32
	MccRiskScore          float32
	MerchantAvgAmount     float32
	MerchantAmountRatio   float32
	HourOfDay             float32
	IsOnline              float32
	CardPresent           float32
	LastKmFromCurrent     float32
	LastTimeDeltaSec      float32
	AmountOverMax         float32
	InstallmentsNorm      float32
	TxVelocity            float32
	IsSafeMCC             float32
	IsRiskyMCC            float32
	AmountNormalized      float32
	CustomerAvgNormalized float32
}

var rawSafeMCCs = map[string]struct{}{
	"5411": {}, "5812": {}, "5912": {}, "5311": {},
}

var rawRiskyMCCs = map[string]struct{}{
	"7995": {}, "7801": {}, "7802": {},
}

func clampF32(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// LastKmFromCurrent and LastTimeDeltaSec are -1.0 when last_transaction is nil.
func ExtractRawFeatures(req dto.FraudRequest, mccRisk map[string]float32) RawFeatures {
	tx := req.Transaction
	cust := req.Customer
	merch := req.Merchant
	term := req.Terminal

	amount := float32(tx.Amount)
	custAvg := float32(cust.AvgAmount)
	merchantAvg := float32(merch.AvgAmount)

	var amountRatio float32
	if custAvg != 0 {
		amountRatio = amount / custAvg
	}

	var merchantAmountRatio float32
	if merchantAvg != 0 {
		merchantAmountRatio = amount / merchantAvg
	}

	isKnown := float32(0)
	for _, m := range cust.KnownMerchants {
		if m == merch.ID {
			isKnown = 1
			break
		}
	}

	mccScore := float32(0.5)
	if score, ok := mccRisk[merch.MCC]; ok {
		mccScore = score
	}

	t, _ := time.Parse(time.RFC3339, tx.RequestedAt)
	t = t.UTC()
	hourOfDay := float32(t.Hour())

	var isOnline, cardPresent float32
	if term.IsOnline {
		isOnline = 1
	}
	if term.CardPresent {
		cardPresent = 1
	}

	lastKm := float32(-1)
	lastDelta := float32(-1)
	if req.LastTx != nil {
		lastKm = float32(req.LastTx.KmFromCurrent)
		lastT, _ := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		lastDelta = float32(math.Abs(t.Sub(lastT).Seconds()))
	}

	var amountOverMax float32
	if amount > 10000 {
		amountOverMax = 1
	}

	_, isSafe := rawSafeMCCs[merch.MCC]
	_, isRisky := rawRiskyMCCs[merch.MCC]

	var isSafeMCC, isRiskyMCC float32
	if isSafe {
		isSafeMCC = 1
	}
	if isRisky {
		isRiskyMCC = 1
	}

	return RawFeatures{
		Amount:                amount,
		AmountRatio:           amountRatio,
		Installments:          float32(tx.Installments),
		TxCount24h:            float32(cust.TxCount24h),
		KmFromHome:            float32(term.KmFromHome),
		IsKnownMerchant:       isKnown,
		MccRiskScore:          mccScore,
		MerchantAvgAmount:     merchantAvg,
		MerchantAmountRatio:   merchantAmountRatio,
		HourOfDay:             hourOfDay,
		IsOnline:              isOnline,
		CardPresent:           cardPresent,
		LastKmFromCurrent:     lastKm,
		LastTimeDeltaSec:      lastDelta,
		AmountOverMax:         amountOverMax,
		InstallmentsNorm:      float32(tx.Installments) / 12.0,
		TxVelocity:            float32(cust.TxCount24h) / 24.0,
		IsSafeMCC:             isSafeMCC,
		IsRiskyMCC:            isRiskyMCC,
		AmountNormalized:      clampF32(amount / 10000.0),
		CustomerAvgNormalized: clampF32(custAvg / 5000.0),
	}
}
