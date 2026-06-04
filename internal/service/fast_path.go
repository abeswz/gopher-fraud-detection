package service

import "gopher-fraud-detection/internal/dto"

var safeMCCs = map[string]struct{}{
	"5411": {}, "5812": {}, "5912": {}, "5311": {},
}

var riskyMCCs = map[string]struct{}{
	"7995": {}, "7801": {}, "7802": {},
}

func fastPath(req dto.FraudRequest) (int, bool) {
	tx := req.Transaction
	cust := req.Customer
	merch := req.Merchant
	term := req.Terminal

	isKnown := false
	for _, m := range cust.KnownMerchants {
		if m == merch.ID {
			isKnown = true
			break
		}
	}

	_, isSafe := safeMCCs[merch.MCC]
	if tx.Amount <= 500 &&
		tx.Amount <= 0.5*cust.AvgAmount &&
		tx.Installments <= 3 &&
		cust.TxCount24h <= 5 &&
		isKnown &&
		term.KmFromHome <= 50 &&
		isSafe {
		return 0, true
	}

	_, isRisky := riskyMCCs[merch.MCC]
	if tx.Amount >= 5000 &&
		tx.Installments >= 5 &&
		cust.TxCount24h >= 6 &&
		!isKnown &&
		term.KmFromHome >= 150 &&
		isRisky {
		return 5, true
	}

	return 0, false
}
