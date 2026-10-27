// Package micrograd implements a scalar automatic differentiation engine.
package micrograd

import "math"

type Operation uint8

const (
	OpNone Operation = iota
	OpAdd
	OpNeg
	OpSub
	OpMul
	OpPow
	OpDiv
	OpExp
	OpLog
	OpTanh
	OpRelu
)

func (o Operation) String() string {
	switch o {
	case OpNone:
		return "OpNone"
	case OpAdd:
		return "OpAdd"
	case OpNeg:
		return "OpNeg"
	case OpSub:
		return "OpSub"
	case OpMul:
		return "OpMul"
	case OpPow:
		return "OpPow"
	case OpDiv:
		return "OpDiv"
	case OpExp:
		return "OpExp"
	case OpLog:
		return "OpLog"
	case OpTanh:
		return "OpTanh"
	case OpRelu:
		return "OpRelu"
	default:
		return "Invalid"
	}
}

// Value represents a single scalar value in the DAG.
type Value struct {
	Data     float64
	Grad     float64 // Grad is the partial derivative of L w.r.t. Value
	prev     [2]*Value
	Op       Operation
	backward func() // backward updates Grad using chain rule
}

func (v *Value) propagate() {
	if v.backward != nil {
		v.backward()
	}
}

func (v *Value) Add(u *Value) *Value {
	out := &Value{
		Data: v.Data + u.Data,
		prev: [2]*Value{v, u},
		Op:   OpAdd,
	}

	out.backward = func() {
		// out = v + u
		// d_out/dv = 1 => dL/dv = dL/d_out * d_out/dv = dL/d_out * 1
		u.Grad += out.Grad
		v.Grad += out.Grad
	}

	return out
}

func (v *Value) Neg() *Value {
	out := &Value{
		Data: v.Data * -1,
		prev: [2]*Value{v},
		Op:   OpNeg,
	}

	out.backward = func() {
		// out = -v
		// d_out/dv = -1 => dL/dv = dL/d_out * d_out/dv = dL/d_out * -1
		v.Grad += out.Grad * -1
	}

	return out
}

func (v *Value) Sub(u *Value) *Value {
	out := &Value{
		Data: v.Data - u.Data,
		prev: [2]*Value{v, u},
		Op:   OpSub,
	}

	out.backward = func() {
		// out = v - u
		// d_out/dv = 1 => dL/dv = dL/d_out * d_out/dv = dL/d_out * 1
		// d_out/du = -1 => dL/dv = dL/d_out * d_out/dv = dL/d_out * -1
		v.Grad += out.Grad
		u.Grad += out.Grad * -1
	}

	return out
}

func (v *Value) Mul(u *Value) *Value {
	out := &Value{
		Data: v.Data * u.Data,
		prev: [2]*Value{v, u},
		Op:   OpMul,
	}

	out.backward = func() {
		// out = v * u => d_out/dv = u
		// dL/dv = dL/d_out * d_out/dv = dL/d_out * u
		v.Grad += out.Grad * u.Data
		u.Grad += out.Grad * v.Data
	}

	return out
}

func (v *Value) Pow(x float64) *Value {
	out := &Value{
		Data: math.Pow(v.Data, x),
		prev: [2]*Value{v},
		Op:   OpPow,
	}

	out.backward = func() {
		// out = v^x => d_out/dv = x(v^x-1)
		// dL/dv = dL/d_out * d_out/dv = dL/d_out * x(v^x-1)
		v.Grad += out.Grad * x * math.Pow(v.Data, x-1)
	}

	return out
}

func (v *Value) Div(u *Value) *Value {
	out := &Value{
		Data: v.Data / u.Data,
		prev: [2]*Value{v, u},
		Op:   OpDiv,
	}

	out.backward = func() {
		// out = v / u
		// d_out/dv = 1/u => dL/dv = dL/d_out * 1/u
		// d_out/du = -v/u^2 = -out/u => dL/du = dL/d_out * -out/u
		v.Grad += out.Grad / u.Data
		u.Grad += out.Grad * -out.Data / u.Data
	}

	return out
}

func (v *Value) Exp() *Value {
	out := &Value{
		Data: math.Exp(v.Data),
		prev: [2]*Value{v},
		Op:   OpExp,
	}

	out.backward = func() {
		// out = e^v => d_out/dv = e^v = out
		// dL/dv = dL/d_out * out
		v.Grad += out.Grad * out.Data
	}

	return out
}

// Log is the natural logarithm. Data <= 0 yields NaN or -Inf, like math.Log.
func (v *Value) Log() *Value {
	out := &Value{
		Data: math.Log(v.Data),
		prev: [2]*Value{v},
		Op:   OpLog,
	}

	out.backward = func() {
		// out = ln(v) => d_out/dv = 1/v
		// dL/dv = dL/d_out * 1/v
		v.Grad += out.Grad / v.Data
	}

	return out
}

func (v *Value) Tanh() *Value {
	out := &Value{
		Data: math.Tanh(v.Data),
		prev: [2]*Value{v},
		Op:   OpTanh,
	}

	out.backward = func() {
		// out = tanh(v)
		// dL/dv = dL/d_out * d_out/dv = dL/d_out * (1 - tanh^2 v)
		v.Grad += out.Grad * (1 - math.Pow(out.Data, 2.0))
	}

	return out
}

func (v *Value) Relu() *Value {
	out := &Value{
		Data: math.Max(0, v.Data),
		prev: [2]*Value{v},
		Op:   OpRelu,
	}

	out.backward = func() {
		// out = max(0, v)
		// dL/dv = dL/d_out * d_out/dv = dL/d_out * (1 if v > 0 else 0)
		if v.Data > 0 {
			v.Grad += out.Grad
		}
	}

	return out
}
