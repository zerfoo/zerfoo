package scheduler

// WarmupLR returns the effective learning rate for the given epoch,
// applying linear warmup over the first warmupEpochs epochs.
func WarmupLR(baseLR float64, epoch, warmupEpochs int) float64 {
	if warmupEpochs <= 0 {
		return baseLR
	}
	scale := float64(epoch+1) / float64(warmupEpochs)
	if scale > 1.0 {
		scale = 1.0
	}
	return baseLR * scale
}
