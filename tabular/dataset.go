package tabular

import (
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// DatasetOptions explicitly identifies columns and the split policy. Features
// default to all columns other than target/group/time; their order is preserved.
// Split is stratified, group, or temporal. Temporal timestamps use RFC3339Nano.
type DatasetOptions struct {
	Target   string   `json:"target"`
	Features []string `json:"features,omitempty"`
	Group    string   `json:"group,omitempty"`
	Time     string   `json:"time,omitempty"`
	Split    string   `json:"split"`
	Seed     uint64   `json:"seed"`
	// Assignments optionally pins 1-based row IDs instead of generating a split.
	Assignments map[string][]int `json:"assignments,omitempty"`
}

// Standardization contains training-only population moments.
type Standardization struct {
	Mean  []float64 `json:"mean"`
	Scale []float64 `json:"scale"`
}

// DatasetManifest identifies source bytes, schema, labels, splits and fitted
// preprocessing. Its hash changes when any of these run inputs changes.
type DatasetManifest struct {
	Version       int              `json:"version"`
	ContentSHA256 string           `json:"content_sha256"`
	Options       DatasetOptions   `json:"options"`
	Labels        []string         `json:"labels"`
	Rows          int              `json:"rows"`
	Assignments   map[string][]int `json:"assignments"`
	Preprocessing Standardization  `json:"preprocessing"`
}

// Dataset holds inspected immutable-by-convention raw numeric rows. Methods
// return copies; source data and manifest cannot be changed through callers.
type Dataset struct {
	manifest      DatasetManifest
	rows          [][]float64
	labels        []int
	groups        []string
	times         []time.Time
	duplicateKeys []string
}

// InspectCSV validates numeric data and prepares deterministic, isolated splits.
// The input is bounded at 64 MiB and nonfinite/float32-overflowing values fail.
func InspectCSV(ctx context.Context, input io.Reader, options DatasetOptions) (*Dataset, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	raw, err := io.ReadAll(io.LimitReader(input, (64<<20)+1))
	if err != nil {
		return nil, fmt.Errorf("tabular: read CSV: %w", err)
	}
	if len(raw) > 64<<20 {
		return nil, fmt.Errorf("tabular: CSV exceeds 64 MiB")
	}
	reader := csv.NewReader(strings.NewReader(string(raw)))
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("tabular: parse CSV: %w", err)
	}
	if len(records) < 4 {
		return nil, fmt.Errorf("tabular: CSV needs a header and at least 3 data rows")
	}
	indexes := make(map[string]int, len(records[0]))
	for i, name := range records[0] {
		if name == "" || !utf8.ValidString(name) {
			return nil, fmt.Errorf("tabular: empty column name")
		}
		if _, ok := indexes[name]; ok {
			return nil, fmt.Errorf("tabular: duplicate column %q", name)
		}
		indexes[name] = i
	}
	if _, ok := indexes[options.Target]; !ok {
		return nil, fmt.Errorf("tabular: target column %q missing", options.Target)
	}
	if options.Split != "stratified" && options.Split != "group" && options.Split != "temporal" {
		return nil, fmt.Errorf("tabular: split must be stratified, group or temporal")
	}
	if options.Split == "group" && options.Group == "" {
		return nil, fmt.Errorf("tabular: group split requires group column")
	}
	if options.Split == "temporal" && options.Time == "" {
		return nil, fmt.Errorf("tabular: temporal split requires time column")
	}
	if options.Group != "" && options.Split != "group" {
		return nil, fmt.Errorf("tabular: group column requires group split; combined group/time policies are unsupported")
	}
	if options.Time != "" && options.Split != "temporal" {
		return nil, fmt.Errorf("tabular: time column requires temporal split")
	}
	for _, column := range []string{options.Group, options.Time} {
		if column != "" {
			if _, ok := indexes[column]; !ok {
				return nil, fmt.Errorf("tabular: column %q missing", column)
			}
			if column == options.Target {
				return nil, fmt.Errorf("tabular: target cannot be group/time column")
			}
		}
	}
	if len(options.Features) == 0 {
		for _, name := range records[0] {
			if name != options.Target && name != options.Group && name != options.Time {
				options.Features = append(options.Features, name)
			}
		}
	}
	seen := make(map[string]bool)
	for _, name := range options.Features {
		if _, ok := indexes[name]; !ok || seen[name] || name == options.Target || name == options.Group || name == options.Time {
			return nil, fmt.Errorf("tabular: invalid, duplicate or leaking feature %q", name)
		}
		seen[name] = true
	}
	if len(options.Features) == 0 {
		return nil, fmt.Errorf("tabular: no numeric features")
	}
	d := &Dataset{}
	names := make([]string, len(records)-1)
	labelSet := make(map[string]bool)
	for i, record := range records[1:] {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		name := record[indexes[options.Target]]
		if strings.TrimSpace(name) == "" || !utf8.ValidString(name) {
			return nil, fmt.Errorf("tabular: row %d missing label", i+1)
		}
		names[i] = name
		labelSet[name] = true
		values := make([]float64, len(options.Features))
		for j, feature := range options.Features {
			value, parseErr := strconv.ParseFloat(record[indexes[feature]], 64)
			if parseErr != nil || math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > math.MaxFloat32 {
				return nil, fmt.Errorf("tabular: row %d feature %q must be finite numeric float32-compatible data", i+1, feature)
			}
			values[j] = value
		}
		d.rows = append(d.rows, values)
		key, marshalErr := json.Marshal(struct {
			Values []float64
			Label  string
		}{values, name})
		if marshalErr != nil {
			return nil, fmt.Errorf("tabular: row identity: %w", marshalErr)
		}
		d.duplicateKeys = append(d.duplicateKeys, string(key))
		if options.Group != "" {
			group := record[indexes[options.Group]]
			if group == "" {
				return nil, fmt.Errorf("tabular: row %d missing group", i+1)
			}
			d.groups = append(d.groups, group)
		}
		if options.Time != "" {
			stamp, parseErr := time.Parse(time.RFC3339Nano, record[indexes[options.Time]])
			if parseErr != nil {
				return nil, fmt.Errorf("tabular: row %d timestamp: %w", i+1, parseErr)
			}
			d.times = append(d.times, stamp)
		}
	}
	labels := make([]string, 0, len(labelSet))
	for name := range labelSet {
		labels = append(labels, name)
	}
	sort.Strings(labels)
	if len(labels) < 2 {
		return nil, fmt.Errorf("tabular: need at least two classes")
	}
	for _, name := range names {
		index, _ := slices.BinarySearch(labels, name)
		d.labels = append(d.labels, index)
	}
	digest := sha256.Sum256(raw)
	d.manifest = DatasetManifest{Version: 1, ContentSHA256: hex.EncodeToString(digest[:]), Options: options, Labels: labels, Rows: len(d.rows)}
	if options.Assignments != nil {
		d.manifest.Assignments = cloneAssignments(options.Assignments)
	} else {
		d.manifest.Assignments = d.makeAssignments()
	}
	if err := d.validateAssignments(); err != nil {
		return nil, err
	}
	d.fitPreprocessing()
	// Do not retain caller-owned slices/maps.
	d.manifest.Options.Features = slices.Clone(options.Features)
	d.manifest.Options.Assignments = nil // Assignments are stored once in the canonical manifest.
	return d, nil
}

func cloneAssignments(source map[string][]int) map[string][]int {
	if source == nil {
		return nil
	}
	result := make(map[string][]int, len(source))
	for name, ids := range source {
		result[name] = slices.Clone(ids)
	}
	return result
}

func (d *Dataset) makeAssignments() map[string][]int {
	assignments := map[string][]int{"train": {}, "validation": {}, "test": {}}
	partitions := []string{"train", "validation", "test"}
	if d.manifest.Options.Split == "temporal" {
		ids := make([]int, len(d.rows))
		for i := range ids {
			ids[i] = i + 1
		}
		sort.SliceStable(ids, func(i, j int) bool { return d.times[ids[i]-1].Before(d.times[ids[j]-1]) })
		trainEnd, valEnd := len(ids)*3/5, len(ids)*4/5
		assignments["train"] = ids[:trainEnd]
		assignments["validation"] = ids[trainEnd:valEnd]
		assignments["test"] = ids[valEnd:]
		return assignments
	}
	groups := make(map[string][]int)
	for i := range d.rows {
		key := d.duplicateKeys[i]
		if d.manifest.Options.Split == "group" {
			key = d.groups[i]
		}
		groups[key] = append(groups[key], i+1)
	}
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		a := sha256.Sum256([]byte(fmt.Sprintf("%d:%s", d.manifest.Options.Seed, keys[i])))
		b := sha256.Sum256([]byte(fmt.Sprintf("%d:%s", d.manifest.Options.Seed, keys[j])))
		return string(a[:]) < string(b[:])
	})
	if d.manifest.Options.Split == "group" {
		// Whole groups follow seeded order. Class coverage is validated afterward;
		// impossible/poorly covered allocations fail rather than moving test rows.
		for i, key := range keys {
			part := 2
			if i < len(keys)*3/5 {
				part = 0
			} else if i < len(keys)*4/5 {
				part = 1
			}
			assignments[partitions[part]] = append(assignments[partitions[part]], groups[key]...)
		}
	} else {
		for class := range d.manifest.Labels {
			total := 0
			for _, label := range d.labels {
				if label == class {
					total++
				}
			}
			caps := []int{total * 3 / 5, total / 5, total - total*3/5 - total/5}
			counts := []int{0, 0, 0}
			for _, key := range keys {
				ids := groups[key]
				if d.labels[ids[0]-1] != class {
					continue
				}
				placed := false
				for part, capacity := range caps {
					if counts[part]+len(ids) <= capacity {
						assignments[partitions[part]] = append(assignments[partitions[part]], ids...)
						counts[part] += len(ids)
						placed = true
						break
					}
				}
				if !placed {
					return nil
				}
			}
		}
	}
	for _, ids := range assignments {
		sort.Ints(ids)
	}
	return assignments
}

func (d *Dataset) validateAssignments() error {
	if len(d.manifest.Assignments) != 3 {
		return fmt.Errorf("tabular: cannot form three isolated splits")
	}
	seen := make(map[int]bool)
	duplicates := make(map[string]string)
	groups := make(map[string]string)
	for _, part := range []string{"train", "validation", "test"} {
		ids := d.manifest.Assignments[part]
		if len(ids) == 0 {
			return fmt.Errorf("tabular: %s split is empty", part)
		}
		classes := make(map[int]bool)
		for _, id := range ids {
			if id < 1 || id > len(d.rows) || seen[id] {
				return fmt.Errorf("tabular: duplicate or invalid row ID %d", id)
			}
			seen[id] = true
			i := id - 1
			classes[d.labels[i]] = true
			key := d.duplicateKeys[i]
			if prior, ok := duplicates[key]; ok && prior != part {
				return fmt.Errorf("tabular: duplicate record crosses %s/%s", prior, part)
			}
			duplicates[key] = part
			if len(d.groups) > 0 {
				group := d.groups[i]
				if prior, ok := groups[group]; ok && prior != part {
					return fmt.Errorf("tabular: group crosses %s/%s", prior, part)
				}
				groups[group] = part
			}
		}
		if len(classes) != len(d.manifest.Labels) {
			return fmt.Errorf("tabular: %s lacks class coverage", part)
		}
	}
	if len(seen) != len(d.rows) {
		return fmt.Errorf("tabular: assignments must cover every row exactly once")
	}
	if len(d.times) > 0 {
		for _, pair := range [][2]string{{"train", "validation"}, {"validation", "test"}} {
			maxBefore := d.times[d.manifest.Assignments[pair[0]][0]-1]
			minAfter := d.times[d.manifest.Assignments[pair[1]][0]-1]
			for _, id := range d.manifest.Assignments[pair[0]] {
				if d.times[id-1].After(maxBefore) {
					maxBefore = d.times[id-1]
				}
			}
			for _, id := range d.manifest.Assignments[pair[1]] {
				if d.times[id-1].Before(minAfter) {
					minAfter = d.times[id-1]
				}
			}
			if !maxBefore.Before(minAfter) {
				return fmt.Errorf("tabular: temporal splits overlap or share a boundary timestamp")
			}
		}
	}
	return nil
}

func (d *Dataset) fitPreprocessing() {
	width := len(d.manifest.Options.Features)
	mean := make([]float64, width)
	scale := make([]float64, width)
	ids := d.manifest.Assignments["train"]
	for _, id := range ids {
		for j, v := range d.rows[id-1] {
			mean[j] += v / float64(len(ids))
		}
	}
	for _, id := range ids {
		for j, v := range d.rows[id-1] {
			delta := v - mean[j]
			scale[j] += delta * delta / float64(len(ids))
		}
	}
	for j := range scale {
		scale[j] = math.Sqrt(scale[j])
		if scale[j] == 0 {
			scale[j] = 1
		}
	}
	d.manifest.Preprocessing = Standardization{Mean: mean, Scale: scale}
}

// Manifest returns an independent copy of the dataset's replay contract.
func (d *Dataset) Manifest() DatasetManifest {
	m := d.manifest
	m.Labels = slices.Clone(m.Labels)
	m.Options.Features = slices.Clone(m.Options.Features)
	m.Options.Assignments = cloneAssignments(m.Options.Assignments)
	m.Assignments = cloneAssignments(m.Assignments)
	m.Preprocessing.Mean = slices.Clone(m.Preprocessing.Mean)
	m.Preprocessing.Scale = slices.Clone(m.Preprocessing.Scale)
	return m
}

// Hash identifies bytes, schema, split assignments and preprocessing together.
func (d *Dataset) Hash() (string, error) {
	raw, err := json.Marshal(d.manifest)
	if err != nil {
		return "", fmt.Errorf("tabular: manifest hash: %w", err)
	}
	digest := sha256.Sum256(raw)
	return hex.EncodeToString(digest[:]), nil
}

// Partition returns copies of standardized features and integer labels.
func (d *Dataset) Partition(name string) ([][]float64, []int, error) {
	ids, ok := d.manifest.Assignments[name]
	if !ok {
		return nil, nil, fmt.Errorf("tabular: unknown partition %q", name)
	}
	rows := make([][]float64, len(ids))
	labels := make([]int, len(ids))
	for i, id := range ids {
		row, err := d.manifest.Preprocessing.Transform(d.rows[id-1])
		if err != nil {
			return nil, nil, err
		}
		rows[i] = row
		labels[i] = d.labels[id-1]
	}
	return rows, labels, nil
}

// Transform applies already-fitted moments; it never learns from input records.
func (s Standardization) Transform(values []float64) ([]float64, error) {
	if len(values) == 0 || len(values) != len(s.Mean) || len(values) != len(s.Scale) {
		return nil, fmt.Errorf("tabular: preprocessing shape mismatch")
	}
	result := make([]float64, len(values))
	for i, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.IsNaN(s.Mean[i]) || math.IsInf(s.Mean[i], 0) || s.Scale[i] <= 0 || math.IsNaN(s.Scale[i]) || math.IsInf(s.Scale[i], 0) {
			return nil, fmt.Errorf("tabular: invalid preprocessing value at feature %d", i)
		}
		result[i] = (v - s.Mean[i]) / s.Scale[i]
		if math.IsNaN(result[i]) || math.IsInf(result[i], 0) || math.Abs(result[i]) > math.MaxFloat32 {
			return nil, fmt.Errorf("tabular: standardized feature %d exceeds float32 range", i)
		}
	}
	return result, nil
}

// WriteManifest persists the replay contract as JSON to the caller's writer.
// A file-owning caller must close and atomically publish its output after success.
func (d *Dataset) WriteManifest(ctx context.Context, output io.Writer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := json.NewEncoder(output).Encode(d.manifest); err != nil {
		return fmt.Errorf("tabular: write dataset manifest: %w", err)
	}
	return nil
}

// ReplayCSV recreates an inspected dataset and rejects changed source bytes,
// labels, schema, split assignments or preprocessing. It never silently refits
// a changed dataset under an existing run's identity.
func ReplayCSV(ctx context.Context, input io.Reader, manifest DatasetManifest) (*Dataset, error) {
	options := manifest.Options
	options.Assignments = cloneAssignments(manifest.Assignments)
	dataset, err := InspectCSV(ctx, input, options)
	if err != nil {
		return nil, err
	}
	expected, err := json.Marshal(manifest)
	if err != nil {
		return nil, fmt.Errorf("tabular: invalid replay manifest: %w", err)
	}
	actual, err := json.Marshal(dataset.manifest)
	if err != nil {
		return nil, fmt.Errorf("tabular: replay manifest: %w", err)
	}
	if string(expected) != string(actual) {
		return nil, fmt.Errorf("tabular: dataset or manifest changed; create a new run")
	}
	return dataset, nil
}
