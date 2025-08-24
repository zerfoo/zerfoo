module github.com/zerfoo/zerfoo

go 1.25

require (
	github.com/parquet-go/parquet-go v0.25.1
	github.com/stretchr/testify v1.11.0
	github.com/zerfoo/float16 v0.1.0
	github.com/zerfoo/float8 v0.2.0
	github.com/zerfoo/zmf v0.1.1
	gonum.org/v1/gonum v0.16.0
	google.golang.org/grpc v1.65.0
	google.golang.org/protobuf v1.36.8
)

require (
	golang.org/x/net v0.25.0 // indirect
	golang.org/x/sys v0.20.0 // indirect
	golang.org/x/text v0.23.0 // indirect
)

exclude google.golang.org/genproto v0.0.0-20220401170504-314d38edb7de

exclude google.golang.org/genproto v0.0.0-20220324131243-acbaeb5b85eb

require (
	gonum.org/v1/gonum v0.16.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240528184218-531527333157 // indirect
)

replace google.golang.org/genproto => google.golang.org/genproto v0.0.0-20240528184218-531527333157
