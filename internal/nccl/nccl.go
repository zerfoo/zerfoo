//go:build cuda

// Package nccl provides CGo bindings for the NVIDIA Collective Communications
// Library (NCCL). It exposes communicator lifecycle, all-reduce, and broadcast
// operations needed for multi-GPU gradient synchronization and tensor
// distribution.
package nccl

/*
#cgo LDFLAGS: -lnccl
#include <nccl.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// DataType specifies the element type for NCCL operations.
type DataType int

const (
	Float32 DataType = C.ncclFloat32
	Float64 DataType = C.ncclFloat64
	Int32   DataType = C.ncclInt32
	Int64   DataType = C.ncclInt64
)

// ReduceOp specifies the reduction operation for collective calls.
type ReduceOp int

const (
	Sum ReduceOp = C.ncclSum
	Avg ReduceOp = C.ncclAvg
	Max ReduceOp = C.ncclMax
	Min ReduceOp = C.ncclMin
)

// UniqueID wraps ncclUniqueId used to bootstrap communicator creation.
type UniqueID struct {
	id C.ncclUniqueId
}

// GetUniqueID generates a new unique ID for communicator initialization.
// Exactly one rank should call this and broadcast the result to all other ranks.
func GetUniqueID() (*UniqueID, error) {
	var id C.ncclUniqueId
	rc := C.ncclGetUniqueId(&id)
	if rc != C.ncclSuccess {
		return nil, fmt.Errorf("ncclGetUniqueId failed: %s", C.GoString(C.ncclGetErrorString(rc)))
	}
	return &UniqueID{id: id}, nil
}

// Bytes returns the raw bytes of the unique ID for serialization.
func (u *UniqueID) Bytes() []byte {
	return C.GoBytes(unsafe.Pointer(&u.id), C.int(unsafe.Sizeof(u.id)))
}

// UniqueIDFromBytes reconstructs a UniqueID from raw bytes.
func UniqueIDFromBytes(b []byte) (*UniqueID, error) {
	var id C.ncclUniqueId
	expected := int(unsafe.Sizeof(id))
	if len(b) != expected {
		return nil, fmt.Errorf("UniqueIDFromBytes: expected %d bytes, got %d", expected, len(b))
	}
	copy((*[1 << 20]byte)(unsafe.Pointer(&id))[:expected], b)
	return &UniqueID{id: id}, nil
}

// Comm wraps an ncclComm_t communicator.
type Comm struct {
	comm C.ncclComm_t
}

// InitRank initializes a communicator for a given rank in a group of nRanks.
// All ranks must call this with the same UniqueID and nRanks. The CUDA device
// for this rank must be set via cuda.SetDevice before calling InitRank.
func InitRank(uid *UniqueID, nRanks, rank int) (*Comm, error) {
	var comm C.ncclComm_t
	rc := C.ncclCommInitRank(&comm, C.int(nRanks), uid.id, C.int(rank))
	if rc != C.ncclSuccess {
		return nil, fmt.Errorf("ncclCommInitRank(nRanks=%d, rank=%d) failed: %s",
			nRanks, rank, C.GoString(C.ncclGetErrorString(rc)))
	}
	return &Comm{comm: comm}, nil
}

// Destroy releases the communicator resources.
func (c *Comm) Destroy() error {
	rc := C.ncclCommDestroy(c.comm)
	if rc != C.ncclSuccess {
		return fmt.Errorf("ncclCommDestroy failed: %s", C.GoString(C.ncclGetErrorString(rc)))
	}
	return nil
}

// AllReduce performs an in-place all-reduce across all ranks. sendBuf and
// recvBuf may be the same pointer for in-place operation. count is the number
// of elements (not bytes). The stream parameter is a cudaStream_t as
// unsafe.Pointer.
func (c *Comm) AllReduce(sendBuf, recvBuf unsafe.Pointer, count int, dtype DataType, op ReduceOp, stream unsafe.Pointer) error {
	rc := C.ncclAllReduce(
		sendBuf,
		recvBuf,
		C.size_t(count),
		C.ncclDataType_t(dtype),
		C.ncclRedOp_t(op),
		c.comm,
		C.cudaStream_t(stream),
	)
	if rc != C.ncclSuccess {
		return fmt.Errorf("ncclAllReduce failed: %s", C.GoString(C.ncclGetErrorString(rc)))
	}
	return nil
}

// Broadcast sends count elements from root's sendBuf to all ranks' recvBuf.
// For root, sendBuf and recvBuf may differ or be the same.
func (c *Comm) Broadcast(sendBuf, recvBuf unsafe.Pointer, count int, dtype DataType, root int, stream unsafe.Pointer) error {
	rc := C.ncclBroadcast(
		sendBuf,
		recvBuf,
		C.size_t(count),
		C.ncclDataType_t(dtype),
		C.int(root),
		c.comm,
		C.cudaStream_t(stream),
	)
	if rc != C.ncclSuccess {
		return fmt.Errorf("ncclBroadcast failed: %s", C.GoString(C.ncclGetErrorString(rc)))
	}
	return nil
}

// GroupStart begins a group of NCCL operations. All NCCL calls between
// GroupStart and GroupEnd are batched into a single launch.
func GroupStart() error {
	rc := C.ncclGroupStart()
	if rc != C.ncclSuccess {
		return fmt.Errorf("ncclGroupStart failed: %s", C.GoString(C.ncclGetErrorString(rc)))
	}
	return nil
}

// GroupEnd completes a group of NCCL operations and launches them.
func GroupEnd() error {
	rc := C.ncclGroupEnd()
	if rc != C.ncclSuccess {
		return fmt.Errorf("ncclGroupEnd failed: %s", C.GoString(C.ncclGetErrorString(rc)))
	}
	return nil
}

// GetAsyncError queries the communicator for any asynchronous errors that
// occurred during previous operations.
func (c *Comm) GetAsyncError() error {
	var result C.ncclResult_t
	rc := C.ncclCommGetAsyncError(c.comm, &result)
	if rc != C.ncclSuccess {
		return fmt.Errorf("ncclCommGetAsyncError query failed: %s", C.GoString(C.ncclGetErrorString(rc)))
	}
	if result != C.ncclSuccess {
		return fmt.Errorf("NCCL async error: %s", C.GoString(C.ncclGetErrorString(result)))
	}
	return nil
}
