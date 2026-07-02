package repository

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"
)

// Handler provides HTTP handlers for the model repository.
type Handler struct {
	repo ModelRepository
}

// NewHandler creates a new Handler backed by the given repository.
func NewHandler(repo ModelRepository) *Handler {
	return &Handler{repo: repo}
}

// RegisterRoutes registers the repository HTTP routes on the given mux.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/models", h.handleList)
	mux.HandleFunc("GET /v1/models/{id}", h.handleGet)
	mux.HandleFunc("POST /v1/models", h.handleUpload)
	mux.HandleFunc("DELETE /v1/models/{id}", h.handleDelete)
}

// listResponse is the response for GET /v1/models.
type listResponse struct {
	Object string          `json:"object"`
	Data   []ModelMetadata `json:"data"`
}

func (h *Handler) handleList(w http.ResponseWriter, _ *http.Request) {
	models, err := h.repo.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal server error")
		return
	}
	if models == nil {
		models = []ModelMetadata{}
	}
	writeJSON(w, http.StatusOK, listResponse{
		Object: "list",
		Data:   models,
	})
}

func (h *Handler) handleGet(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	meta, err := h.repo.Get(id)
	if err != nil {
		if errors.Is(err, ErrPathTraversal) {
			writeError(w, http.StatusBadRequest, "invalid model id")
			return
		}
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "model '"+id+"' not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "internal server error")
		return
	}
	writeJSON(w, http.StatusOK, meta)
}

// uploadRequest is the JSON body for POST /v1/models when not using multipart.
type uploadRequest struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Version string `json:"version"`
	Format  string `json:"format"`
}

func (h *Handler) handleUpload(w http.ResponseWriter, r *http.Request) {
	// Support multipart upload: metadata in "metadata" field, model file in "file" field.
	ct := r.Header.Get("Content-Type")
	if ct == "" || !isMultipart(ct) {
		writeError(w, http.StatusBadRequest, "Content-Type must be multipart/form-data")
		return
	}

	// Limit to 10 GB.
	const maxUpload = 10 << 30
	r.Body = http.MaxBytesReader(w, r.Body, maxUpload)

	if err := r.ParseMultipartForm(32 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "failed to parse multipart form: "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()

	metaJSON := r.FormValue("metadata")
	if metaJSON == "" {
		writeError(w, http.StatusBadRequest, "missing 'metadata' form field")
		return
	}

	var req uploadRequest
	if err := json.Unmarshal([]byte(metaJSON), &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid metadata JSON: "+err.Error())
		return
	}

	if req.ID == "" {
		writeError(w, http.StatusBadRequest, "model id is required")
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "model name is required")
		return
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing 'file' form field: "+err.Error())
		return
	}
	defer file.Close()

	meta := ModelMetadata{
		ID:        req.ID,
		Name:      req.Name,
		Version:   req.Version,
		Format:    req.Format,
		CreatedAt: time.Now().UTC(),
	}

	if err := h.repo.Upload(meta, file); err != nil {
		if errors.Is(err, ErrPathTraversal) {
			writeError(w, http.StatusBadRequest, "invalid model id")
			return
		}
		if errors.Is(err, ErrAlreadyExists) {
			writeError(w, http.StatusConflict, "model '"+req.ID+"' already exists")
			return
		}
		writeError(w, http.StatusInternalServerError, "internal server error")
		return
	}

	// Re-read metadata to get computed fields (SHA256, size).
	stored, err := h.repo.Get(req.ID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal server error")
		return
	}
	writeJSON(w, http.StatusCreated, stored)
}

// deleteResponse is the response for DELETE /v1/models/:id.
type deleteResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

func (h *Handler) handleDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.repo.Delete(id); err != nil {
		if errors.Is(err, ErrPathTraversal) {
			writeError(w, http.StatusBadRequest, "invalid model id")
			return
		}
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "model '"+id+"' not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "internal server error")
		return
	}
	writeJSON(w, http.StatusOK, deleteResponse{
		ID:      id,
		Object:  "model",
		Deleted: true,
	})
}

func isMultipart(ct string) bool {
	// Simple prefix check; the boundary parameter follows.
	return len(ct) >= 19 && ct[:19] == "multipart/form-data"
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]interface{}{
		"error": map[string]string{"message": message},
	})
}

// ReadCloserFunc wraps an io.Reader with a no-op Close for use as io.ReadCloser.
type ReadCloserFunc struct {
	io.Reader
}

// Close is a no-op.
func (ReadCloserFunc) Close() error { return nil }
