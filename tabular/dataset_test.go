package tabular

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestDatasetPinnedIris(t *testing.T) {
	var fixture struct {
		SHA256 map[string]string `json:"sha256"`
		Splits map[string][]int  `json:"splits"`
	}
	raw, err := os.ReadFile("testdata/model_creation/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	csv, err := os.ReadFile("testdata/model_creation/iris.csv")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(csv)
	if hex.EncodeToString(sum[:]) != fixture.SHA256["iris.csv"] {
		t.Fatal("fixture hash mismatch")
	}
	options := DatasetOptions{Target: "species", Split: "stratified", Assignments: fixture.Splits}
	data, err := InspectCSV(context.Background(), strings.NewReader(string(csv)), options)
	if err != nil {
		t.Fatal(err)
	}
	if data.Manifest().Rows != 150 {
		t.Fatal("wrong row count")
	}
	for part, count := range map[string]int{"train": 90, "validation": 30, "test": 30} {
		rows, labels, err := data.Partition(part)
		if err != nil {
			t.Fatal(err)
		}
		if len(rows) != count || len(labels) != count {
			t.Fatal("partition count")
		}
	}
	// Every fitted training feature has zero mean after standardization.
	train, _, err := data.Partition("train")
	if err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 4; j++ {
		sum := 0.0
		for _, row := range train {
			sum += row[j]
		}
		if sum > 1e-10 || sum < -1e-10 {
			t.Fatalf("not centered: %g", sum)
		}
	}
	// Changing the caller's pinned assignments cannot mutate the dataset.
	before, err := data.Hash()
	if err != nil {
		t.Fatal(err)
	}
	options.Assignments["train"][0] = 999
	after, err := data.Hash()
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatal("caller changed manifest")
	}
}

func datasetCSV() string {
	var b strings.Builder
	b.WriteString("x,group,time,label\n")
	for i := 0; i < 30; i++ {
		fmt.Fprintf(&b, "%d,g%d,%s,%s\n", i, i/2, time.Date(2026, 1, 1, 0, i, 0, 0, time.UTC).Format(time.RFC3339), []string{"no", "yes"}[i%2])
	}
	return b.String()
}

func TestDatasetSplitReplayAndIsolation(t *testing.T) {
	for _, mode := range []string{"stratified", "group", "temporal"} {
		t.Run(mode, func(t *testing.T) {
			options := DatasetOptions{Target: "label", Features: []string{"x"}, Split: mode, Seed: 42}
			if mode == "group" {
				options.Group = "group"
			}
			if mode == "temporal" {
				options.Time = "time"
			}
			first, err := InspectCSV(context.Background(), strings.NewReader(datasetCSV()), options)
			if err != nil {
				t.Fatal(err)
			}
			second, err := InspectCSV(context.Background(), strings.NewReader(datasetCSV()), options)
			if err != nil {
				t.Fatal(err)
			}
			firstHash, err := first.Hash()
			if err != nil {
				t.Fatal(err)
			}
			secondHash, err := second.Hash()
			if err != nil {
				t.Fatal(err)
			}
			if firstHash != secondHash {
				t.Fatal("nondeterministic split")
			}
			options.Assignments = first.Manifest().Assignments
			replay, err := InspectCSV(context.Background(), strings.NewReader(datasetCSV()), options)
			if err != nil {
				t.Fatal(err)
			}
			replayHash, err := replay.Hash()
			if err != nil {
				t.Fatal(err)
			}
			if replayHash != firstHash {
				t.Fatal("pinned replay changed identity")
			}
			// Alter just a held-out feature. The data/run hash changes, training moments do not.
			id := first.Manifest().Assignments["test"][0]
			lines := strings.Split(datasetCSV(), "\n")
			fields := strings.Split(lines[id], ",")
			fields[0] = "1000000"
			lines[id] = strings.Join(fields, ",")
			outlier, err := InspectCSV(context.Background(), strings.NewReader(strings.Join(lines, "\n")), options)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(first.Manifest().Preprocessing, outlier.Manifest().Preprocessing) {
				t.Fatal("holdout leaked into preprocessing")
			}
			changed, err := outlier.Hash()
			if err != nil {
				t.Fatal(err)
			}
			if changed == firstHash {
				t.Fatal("changed data retained identity")
			}
			var saved bytes.Buffer
			if err := first.WriteManifest(context.Background(), &saved); err != nil {
				t.Fatal(err)
			}
			var manifest DatasetManifest
			if err := json.Unmarshal(saved.Bytes(), &manifest); err != nil {
				t.Fatal(err)
			}
			if _, err := ReplayCSV(context.Background(), strings.NewReader(datasetCSV()), manifest); err != nil {
				t.Fatal(err)
			}
			if _, err := ReplayCSV(context.Background(), strings.NewReader(strings.Join(lines, "\n")), manifest); err == nil {
				t.Fatal("changed source accepted on resume")
			}

		})
	}
}

func TestDatasetRejectsInvalidContracts(t *testing.T) {
	base := DatasetOptions{Target: "label", Features: []string{"x"}, Split: "temporal", Time: "time"}
	original, err := InspectCSV(context.Background(), strings.NewReader(datasetCSV()), base)
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*DatasetOptions){
		"unknown split":        func(o *DatasetOptions) { o.Split = "random" },
		"missing target":       func(o *DatasetOptions) { o.Target = "missing" },
		"target leakage":       func(o *DatasetOptions) { o.Features = []string{"label"} },
		"duplicate features":   func(o *DatasetOptions) { o.Features = []string{"x", "x"} },
		"group without column": func(o *DatasetOptions) { o.Split = "group"; o.Time = "" },
		"row overlap": func(o *DatasetOptions) {
			o.Assignments = original.Manifest().Assignments
			o.Assignments["test"][0] = o.Assignments["train"][0]
		},
		"future training": func(o *DatasetOptions) {
			o.Assignments = original.Manifest().Assignments
			o.Assignments["test"][0], o.Assignments["train"][0] = o.Assignments["train"][0], o.Assignments["test"][0]
		},
	} {
		t.Run(name, func(t *testing.T) {
			options := base
			mutate(&options)
			if _, err := InspectCSV(context.Background(), strings.NewReader(datasetCSV()), options); err == nil {
				t.Fatal("invalid contract accepted")
			}
		})
	}
	for name, replace := range map[string][2]string{
		"nonfinite":        {"0,g0,", "NaN,g0,"},
		"overflow":         {"0,g0,", "1e100,g0,"},
		"missing numeric":  {"0,g0,", ",g0,"},
		"duplicate header": {"x,group,time,label", "x,x,time,label"},
		"unseen class":     {"00:29:00Z,yes", "00:29:00Z,newclass"},
		"missing label":    {"00:29:00Z,yes", "00:29:00Z,"},
		"invalid time":     {"00:29:00Z", "bad"},
	} {
		t.Run(name, func(t *testing.T) {
			raw := strings.Replace(datasetCSV(), replace[0], replace[1], 1)
			if _, err := InspectCSV(context.Background(), strings.NewReader(raw), base); err == nil {
				t.Fatal("malformed data accepted")
			}
		})
	}
	// Make the last record duplicate a training record's features and label.
	raw := strings.Replace(datasetCSV(), "29,g14,", "1,g14,", 1)
	if _, err := InspectCSV(context.Background(), strings.NewReader(raw), base); err == nil {
		t.Fatal("duplicate record crossed split")
	}
	grouped := DatasetOptions{Target: "label", Features: []string{"x"}, Split: "group", Group: "group"}
	groupData, err := InspectCSV(context.Background(), strings.NewReader(datasetCSV()), grouped)
	if err != nil {
		t.Fatal(err)
	}
	grouped.Assignments = groupData.Manifest().Assignments
	grouped.Assignments["train"][0], grouped.Assignments["test"][0] = grouped.Assignments["test"][0], grouped.Assignments["train"][0]
	if _, err := InspectCSV(context.Background(), strings.NewReader(datasetCSV()), grouped); err == nil {
		t.Fatal("group crossing accepted")
	}
}
