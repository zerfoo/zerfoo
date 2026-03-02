package config

// EngineConfig holds configuration for the compute engine.
type EngineConfig struct {
	Device        string `env:"DEVICE"          json:"device"          validate:"required"`
	MemoryLimitMB int    `env:"MEMORY_LIMIT_MB" json:"memory_limit_mb"`
	LogLevel      string `env:"LOG_LEVEL"       json:"log_level"`
}

// TrainingConfig holds configuration for training workflows.
type TrainingConfig struct {
	BatchSize          int    `env:"BATCH_SIZE"          json:"batch_size"          validate:"required"`
	LearningRate       string `env:"LEARNING_RATE"       json:"learning_rate"       validate:"required"`
	Optimizer          string `env:"OPTIMIZER"           json:"optimizer"           validate:"required"`
	Epochs             int    `env:"EPOCHS"              json:"epochs"`
	CheckpointInterval int    `env:"CHECKPOINT_INTERVAL" json:"checkpoint_interval"`
}

// DistributedConfig holds configuration for distributed training.
type DistributedConfig struct {
	CoordinatorAddress string `env:"COORDINATOR_ADDRESS" json:"coordinator_address" validate:"required"`
	TimeoutSeconds     int    `env:"TIMEOUT_SECONDS"     json:"timeout_seconds"`
	TLSEnabled         bool   `env:"TLS_ENABLED"         json:"tls_enabled"`
}
