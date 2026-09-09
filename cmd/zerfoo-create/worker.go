package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	bolt "go.etcd.io/bbolt"
)

func (s *service) spawn(id string) error {
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	log, err := os.OpenFile(filepath.Join(s.state, id+".log"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	command := exec.CommandContext(context.Background(), executable, "--state", s.state, "worker", id)
	command.Stdout = log
	command.Stderr = log
	if err := command.Start(); err != nil {
		return errors.Join(err, log.Close())
	}
	return errors.Join(command.Process.Release(), log.Close())
}
func (s *service) work(ctx context.Context, id string) (err error) {
	var r run
	if err := s.load("run/"+id, &r); err != nil {
		return err
	}
	// Only one worker computes per installation. The OS releases this lock on exit.
	lease, err := bolt.Open(filepath.Join(s.state, "worker.db"), 0600, &bolt.Options{Timeout: time.Second})
	if err != nil {
		return fmt.Errorf("another worker owns this installation: %w", err)
	}
	defer func() { err = errors.Join(err, lease.Close()) }()
	// Re-read after acquiring the worker lock; another attempt may have finished.
	if err := s.load("run/"+id, &r); err != nil {
		return err
	}
	if r.Status != "queued" {
		return fmt.Errorf("run is not queued")
	}
	defer func() {
		if err != nil {
			r.Status = "failed"
			if errors.Is(err, context.Canceled) {
				r.Status = "canceled"
			}
			if errors.Is(err, tabular.ErrTrainingLimit) {
				r.Status = "budget_exhausted"
			}
			r.Error = err.Error()
		}
		r.Updated = time.Now().UTC()
		err = errors.Join(err, s.transaction(func(b *bolt.Bucket) error {
			var current run
			if loadErr := get(b, "run/"+id, &current); loadErr != nil {
				return loadErr
			}
			r.Cancel = current.Cancel
			if r.Cancel && r.Status == "succeeded" {
				r.Status = "canceled"
				r.Error = context.Canceled.Error()
			}
			return put(b, "run/"+id, r)
		}))
	}()
	var p plan
	if err := s.load("plan/"+r.Plan, &p); err != nil {
		return err
	}
	if r.Cancel {
		return context.Canceled
	}
	r.Status = "running"
	r.Updated = time.Now().UTC()
	if err := s.save("run/"+id, r); err != nil {
		return err
	}
	dataset, err := s.dataset(ctx, p.Dataset)
	if err != nil {
		return err
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	result, err := tabular.FitClassifier(ctx, dataset, p.Config, p.Options, engine, func(progress tabular.TrainingProgress) error {
		return s.transaction(func(b *bolt.Bucket) error {
			var current run
			if err := get(b, "run/"+id, &current); err != nil {
				return err
			}
			r.Cancel = current.Cancel
			r.Progress = progress
			r.Updated = time.Now().UTC()
			if r.Cancel {
				return context.Canceled
			}
			return put(b, "run/"+id, r)
		})
	})
	if err != nil {
		return err
	}
	rows, labels, err := dataset.Partition("validation")
	if err != nil {
		return err
	}
	evaluation, err := result.Model.Evaluate(ctx, rows, labels)
	if err != nil {
		return err
	}
	r.Artifact = filepath.Join(s.state, id+".bundle")
	r.Hash, err = tabular.SaveClassifierBundle(ctx, r.Artifact, result.Model, dataset, p.ID)
	if err != nil {
		return err
	}
	r.Validation = &evaluation
	r.Status = "succeeded"
	return nil
}
