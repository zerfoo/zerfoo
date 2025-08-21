module github.com/zerfoo/zerfoo

go 1.25

require (
	github.com/zerfoo/float16 v0.1.0
	github.com/zerfoo/float8 v0.1.1
	github.com/zerfoo/zmf v0.1.0
	google.golang.org/grpc v1.65.0
	google.golang.org/protobuf v1.36.8
)

replace github.com/zerfoo/zmf => ../zmf

require (
	golang.org/x/net v0.25.0 // indirect
	golang.org/x/sys v0.20.0 // indirect
	golang.org/x/text v0.15.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240528184218-531527333157 // indirect
)
