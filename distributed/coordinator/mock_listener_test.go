package coordinator

import "net"

// customMockListener is a mock net.Listener for testing.
type customMockListener struct {
	AcceptErr error
	CloseErr  error
	AddrVal   net.Addr
}

func (m *customMockListener) Accept() (net.Conn, error) { return nil, m.AcceptErr }
func (m *customMockListener) Close() error              { return m.CloseErr }
func (m *customMockListener) Addr() net.Addr            { return m.AddrVal }
