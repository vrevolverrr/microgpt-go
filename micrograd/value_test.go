package micrograd

import (
	"math"
	"slices"
	"testing"
)

func near(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9
}

func TestValue_Add_DistinctOperand(t *testing.T) {
	v := &Value{Data: 3.0}
	u := &Value{Data: 2.0}
	L := u.Add(v)

	if got, want := L.Op, OpAdd; got != want {
		t.Errorf("Op = %s, want %s", got, want)
	}

	if got, want := L.Data, u.Data+v.Data; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = u + v => dL/dv = 1
	grad := 2.0
	L.Grad = grad
	L.propagate()

	for name, child := range map[string]*Value{"v": v, "u": u} {
		if !slices.Contains(L.prev[:], child) {
			t.Errorf("Prev missing %s (Data=%v)", name, child.Data)
		}

		if got, want := child.Grad, grad*1; !near(got, want) {
			t.Errorf("%s.Grad = %v, want %v", name, got, want)
		}
	}
}

func TestValue_Add_SameOperand(t *testing.T) {
	v := &Value{Data: 3.0}
	L := v.Add(v)

	if got, want := L.Data, v.Data*2; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = 2v => dL/dv = 2
	grad := 2.0
	L.Grad = grad
	L.propagate()

	if got, want := v.Grad, grad*2; !near(got, want) {
		t.Errorf("v.Grad = %v, want %v", got, want)
	}
}

func TestValue_Mul_DistinctOperand(t *testing.T) {
	v := &Value{Data: 3.0}
	u := &Value{Data: 2.0}
	L := u.Mul(v)

	if got, want := L.Op, OpMul; got != want {
		t.Errorf("Op = %s, want %s", got, want)
	}

	if got, want := L.Data, u.Data*v.Data; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = u * v => dL/dv = u, dL/du = v
	grad := 2.0
	L.Grad = grad
	L.propagate()

	wantGrads := map[string]float64{"v": grad * u.Data, "u": grad * v.Data}

	for name, child := range map[string]*Value{"v": v, "u": u} {
		if !slices.Contains(L.prev[:], child) {
			t.Errorf("Prev missing %s (Data=%v)", name, child.Data)
		}

		if got, want := child.Grad, wantGrads[name]; !near(got, want) {
			t.Errorf("%s.Grad = %v, want %v", name, got, want)
		}
	}
}

func TestValue_Mul_SameOperand(t *testing.T) {
	v := &Value{Data: 3.0}
	L := v.Mul(v)

	if got, want := L.Data, v.Data*v.Data; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = v^2 => dL/dv = 2v
	grad := 2.0
	L.Grad = grad
	L.propagate()

	if got, want := v.Grad, grad*2*v.Data; !near(got, want) {
		t.Errorf("v.Grad = %v, want %v", got, want)
	}
}

func TestValue_Tanh(t *testing.T) {
	v := &Value{Data: 3.0}
	L := v.Tanh()

	if got, want := L.Op, OpTanh; got != want {
		t.Errorf("Op = %s, want %s", got, want)
	}

	if got, want := L.Data, math.Tanh(v.Data); !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	if !slices.Contains(L.prev[:], v) {
		t.Errorf("Prev missing v (Data=%v)", v.Data)
	}

	// L = tanh(v) => dL/dv = 1 - tanh^2(v)
	grad := 2.0
	L.Grad = grad
	L.propagate()

	if got, want := v.Grad, grad*(1-math.Pow(math.Tanh(v.Data), 2)); !near(got, want) {
		t.Errorf("v.Grad = %v, want %v", got, want)
	}
}

func TestValue_TanhOverflow(t *testing.T) {
	for _, tc := range []struct{ in, want float64 }{{10000, 1}, {-10000, -1}} {
		v := &Value{Data: tc.in}
		L := v.Tanh()

		if got, want := L.Data, tc.want; got != want {
			t.Errorf("Tanh(%v) = %v, want %v", tc.in, got, want)
		}

		L.Grad = 2.0
		L.propagate()

		if got, want := v.Grad, 0.0; got != want {
			t.Errorf("Tanh(%v): v.Grad = %v, want %v", tc.in, got, want)
		}
	}
}

func TestValue_Relu(t *testing.T) {
	for _, tc := range []struct{ in, want, wantGrad float64 }{{-3.0, 0, 0}, {3.0, 3.0, 1.0}} {
		v := &Value{Data: tc.in}
		L := v.Relu()

		if got, want := L.Data, tc.want; got != want {
			t.Errorf("Relu(%v) = %v, want %v", tc.in, got, want)
		}

		// L = Relu(v) => dL/dv = 1 if v > 0 else 0
		L.Grad = 2.0
		L.propagate()

		if got, want := v.Grad, tc.wantGrad*L.Grad; got != want {
			t.Errorf("v.Grad = %v, want %v", got, want)
		}
	}
}

func TestValue_Neg(t *testing.T) {
	v := &Value{Data: 3.0}
	L := v.Neg()

	if got, want := L.Op, OpNeg; got != want {
		t.Errorf("Op = %s, want %s", got, want)
	}

	if got, want := L.Data, -v.Data; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	if !slices.Contains(L.prev[:], v) {
		t.Errorf("Prev missing v (Data=%v)", v.Data)
	}

	// L = -v => dL/dv = -1
	grad := 2.0
	L.Grad = grad
	L.propagate()

	if got, want := v.Grad, grad*-1; !near(got, want) {
		t.Errorf("v.Grad = %v, want %v", got, want)
	}
}

func TestValue_Sub_DistinctOperand(t *testing.T) {
	v := &Value{Data: 3.0}
	u := &Value{Data: 2.0}
	L := v.Sub(u)

	if got, want := L.Op, OpSub; got != want {
		t.Errorf("Op = %s, want %s", got, want)
	}

	if got, want := L.Data, v.Data-u.Data; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = v - u => dL/dv = 1, dL/du = -1
	grad := 2.0
	L.Grad = grad
	L.propagate()

	wantGrads := map[string]float64{"v": grad * 1, "u": grad * -1}

	for name, child := range map[string]*Value{"v": v, "u": u} {
		if !slices.Contains(L.prev[:], child) {
			t.Errorf("Prev missing %s (Data=%v)", name, child.Data)
		}

		if got, want := child.Grad, wantGrads[name]; !near(got, want) {
			t.Errorf("%s.Grad = %v, want %v", name, got, want)
		}
	}
}

func TestValue_Sub_SameOperand(t *testing.T) {
	v := &Value{Data: 3.0}
	L := v.Sub(v)

	if got, want := L.Data, 0.0; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = v - v = 0 => dL/dv = 1 - 1 = 0
	L.Grad = 2.0
	L.propagate()

	if got, want := v.Grad, 0.0; !near(got, want) {
		t.Errorf("v.Grad = %v, want %v", got, want)
	}
}

func TestValue_Pow(t *testing.T) {
	for _, tc := range []struct{ in, x, want, wantGrad float64 }{
		{3.0, 2.0, 9.0, 6.0},    // dL/dv = 2 * 3
		{2.0, -1.0, 0.5, -0.25}, // dL/dv = -1 * 2^-2
		{4.0, 0.5, 2.0, 0.25},   // dL/dv = 0.5 * 4^-0.5
	} {
		v := &Value{Data: tc.in}
		L := v.Pow(tc.x)

		if got, want := L.Op, OpPow; got != want {
			t.Errorf("Op = %s, want %s", got, want)
		}

		if got, want := L.Data, tc.want; !near(got, want) {
			t.Errorf("Pow(%v, %v) = %v, want %v", tc.in, tc.x, got, want)
		}

		if !slices.Contains(L.prev[:], v) {
			t.Errorf("Prev missing v (Data=%v)", v.Data)
		}

		grad := 2.0
		L.Grad = grad
		L.propagate()

		if got, want := v.Grad, tc.wantGrad*grad; !near(got, want) {
			t.Errorf("Pow(%v, %v): v.Grad = %v, want %v", tc.in, tc.x, got, want)
		}
	}
}

func TestValue_Div_DistinctOperand(t *testing.T) {
	v := &Value{Data: 6.0}
	u := &Value{Data: 3.0}
	L := v.Div(u)

	if got, want := L.Op, OpDiv; got != want {
		t.Errorf("Op = %s, want %s", got, want)
	}

	if got, want := L.Data, 2.0; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = v / u => dL/dv = 1/u, dL/du = -v/u^2
	grad := 2.0
	L.Grad = grad
	L.propagate()

	wantGrads := map[string]float64{"v": grad / 3.0, "u": grad * -6.0 / 9.0}

	for name, child := range map[string]*Value{"v": v, "u": u} {
		if !slices.Contains(L.prev[:], child) {
			t.Errorf("Prev missing %s (Data=%v)", name, child.Data)
		}

		if got, want := child.Grad, wantGrads[name]; !near(got, want) {
			t.Errorf("%s.Grad = %v, want %v", name, got, want)
		}
	}
}

func TestValue_Div_SameOperand(t *testing.T) {
	v := &Value{Data: 3.0}
	L := v.Div(v)

	if got, want := L.Data, 1.0; !near(got, want) {
		t.Errorf("Data = %v, want %v", got, want)
	}

	// L = v / v = 1 => dL/dv = 1/v - v/v^2 = 0
	L.Grad = 2.0
	L.propagate()

	if got, want := v.Grad, 0.0; !near(got, want) {
		t.Errorf("v.Grad = %v, want %v", got, want)
	}
}

func TestValue_Exp(t *testing.T) {
	for _, in := range []float64{-2.0, 0.0, 1.5} {
		v := &Value{Data: in}
		L := v.Exp()

		if got, want := L.Op, OpExp; got != want {
			t.Errorf("Op = %s, want %s", got, want)
		}

		if got, want := L.Data, math.Exp(in); !near(got, want) {
			t.Errorf("Exp(%v) = %v, want %v", in, got, want)
		}

		if !slices.Contains(L.prev[:], v) {
			t.Errorf("Prev missing v (Data=%v)", v.Data)
		}

		// L = e^v => dL/dv = e^v
		grad := 2.0
		L.Grad = grad
		L.propagate()

		if got, want := v.Grad, grad*math.Exp(in); !near(got, want) {
			t.Errorf("Exp(%v): v.Grad = %v, want %v", in, got, want)
		}
	}
}

func TestValue_Log(t *testing.T) {
	for _, in := range []float64{0.5, 1.0, 4.0} {
		v := &Value{Data: in}
		L := v.Log()

		if got, want := L.Op, OpLog; got != want {
			t.Errorf("Op = %s, want %s", got, want)
		}

		if got, want := L.Data, math.Log(in); !near(got, want) {
			t.Errorf("Log(%v) = %v, want %v", in, got, want)
		}

		if !slices.Contains(L.prev[:], v) {
			t.Errorf("Prev missing v (Data=%v)", v.Data)
		}

		// L = ln(v) => dL/dv = 1/v
		grad := 2.0
		L.Grad = grad
		L.propagate()

		if got, want := v.Grad, grad/in; !near(got, want) {
			t.Errorf("Log(%v): v.Grad = %v, want %v", in, got, want)
		}
	}
}

func TestValue_Grad_AccumulatesAcrossOps(t *testing.T) {
	x := &Value{Data: 2.0}
	a := x.Tanh()
	y := &Value{Data: 3.0}
	b := a.Mul(y)
	L := a.Add(b)

	// L = tanh(x) + y * tanh(x) = tanh(x) * (y + 1)
	// dL/dx = (y + 1)(1 - tanh^2(x))
	// dL/dy = tanh(x)
	wantX := (y.Data + 1) * (1 - math.Pow(math.Tanh(x.Data), 2))
	wantY := math.Tanh(x.Data)

	// a receives from both L and b, so b must propagate before a.
	L.Grad = 1.0
	L.propagate()
	b.propagate()
	a.propagate()
	x.propagate()
	y.propagate()

	// L = a * (1 + y)
	// dL/da = 1 + y
	if got, want := a.Grad, 1+y.Data; !near(got, want) {
		t.Errorf("a.Grad = %v, want %v", got, want)
	}

	if got, want := x.Grad, wantX; !near(got, want) {
		t.Errorf("x.Grad = %v, want %v", got, want)
	}

	if got, want := y.Grad, wantY; !near(got, want) {
		t.Errorf("y.Grad = %v, want %v", got, want)
	}
}
