package optimizer

import "github.com/zerfoo/zerfoo/model"

func init() {
	// Discovery describes the existing AdamW implementation, not a new optimizer.
	if err := model.RegisterComponent(model.ComponentDescriptor{ID: "adamw", Version: 1, Kind: "optimizer", Support: []model.ExecutionSupport{{Operation: "training", Device: "cpu", Precision: "float32", Mode: "eager", Status: "implemented"}}}, nil); err != nil {
		return
	}
}
