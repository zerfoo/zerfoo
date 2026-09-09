package loss

import "github.com/zerfoo/zerfoo/model"

func init() {
	// A malformed descriptor fails closed: the operation remains undiscoverable.
	if err := model.RegisterComponent(model.ComponentDescriptor{ID: "cross_entropy", Version: 1, Kind: "loss", Support: []model.ExecutionSupport{{Operation: "training", Device: "cpu", Precision: "float32", Mode: "eager", Status: "implemented"}}}, nil); err != nil {
		return
	}
}
