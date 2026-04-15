module github.com/zerfoo/zerfoo

go 1.26.0

require (
	github.com/zerfoo/float16 v0.2.0
	github.com/zerfoo/float8 v0.2.0
	go.etcd.io/bbolt v1.4.3
	golang.org/x/image v0.37.0
	google.golang.org/grpc v1.79.3
	google.golang.org/protobuf v1.36.10
)

require (
	golang.org/x/net v0.48.0 // indirect
	golang.org/x/sys v0.39.0 // indirect
	golang.org/x/text v0.35.0 // indirect
	gonum.org/v1/gonum v0.17.0 // indirect
)

exclude google.golang.org/genproto v0.0.0-20220401170504-314d38edb7de

exclude google.golang.org/genproto v0.0.0-20220324131243-acbaeb5b85eb

require (
	github.com/zerfoo/ztensor v1.5.1-0.20260415020357-6ecf8db03600
	github.com/zerfoo/ztoken v0.3.4
	google.golang.org/genproto/googleapis/rpc v0.0.0-20251202230838-ff82c1b0f217 // indirect
)
