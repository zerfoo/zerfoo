package timeseries

// trainWindowedEngine implements GPU-accelerated iTransformer training.
// This is a stub that will be replaced by the full implementation.
func (m *ITransformer) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	// TODO: T113.2 will implement this.
	// For now, fall back to the CPU path by calling the inline training logic.
	// This should never be called since the dispatch checks engine != nil,
	// and the engine is only set via WithITransformerEngine.
	panic("itransformer: engine training not yet implemented")
}
