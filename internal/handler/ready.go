package handler

import "net/http"

func Ready(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}
