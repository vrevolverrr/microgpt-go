package micrograd

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestBuildTopo(t *testing.T) {
	x := &Value{Data: 2.0}
	a := x.Tanh()
	y := &Value{Data: 3.0}
	b := a.Mul(y)
	L := a.Add(b)

	nodes := map[*Value]string{
		x: "x", a: "a", y: "y", b: "b", L: "L",
	}

	names := func(vs []*Value) []string {
		out := make([]string, len(vs))
		for i, v := range vs {
			out[i] = nodes[v]
		}
		return out
	}

	order := buildTopo(L, map[*Value]struct{}{}, []*Value{})
	if want := []*Value{x, a, y, b, L}; !slices.Equal(order, want) {
		t.Errorf("order = %v, expected: %v", names(order), names(want))
	}
}

func TestBuildTopo_Random(t *testing.T) {
	ops := []func(a, b *Value) *Value{
		func(a, b *Value) *Value { return a.Add(b) },
		func(a, b *Value) *Value { return a.Mul(b) },
		func(a, _ *Value) *Value { return a.Tanh() }, // unary: ignores b
	}

	for seed := uint64(0); seed < 100; seed++ {
		rng := rand.New(rand.NewPCG(seed, 0))
		n := 1 + rng.IntN(30)
		nodes := []*Value{{Data: 1.0}, {Data: 2.0}}

		for i := 0; i < n; i++ {
			a := nodes[rng.IntN(len(nodes))]
			b := nodes[rng.IntN(len(nodes))]
			nodes = append(nodes, ops[rng.IntN(len(ops))](a, b))
		}

		order := buildTopo(nodes[len(nodes)-1], map[*Value]struct{}{}, []*Value{})

		pos := map[*Value]int{}
		for i, v := range order {
			pos[v] = i
		}

		for _, v := range order {
			for _, p := range v.prev {
				if p == nil {
					continue
				}

				if pos[p] >= pos[v] {
					t.Errorf("seed %d: node %d appears before its operand %d", seed, pos[v], pos[p])
				}
			}
		}
	}
}

func TestBackward(t *testing.T) {
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

	// same test as TestValue_Grad_AccumulatesAcrossOps, but replace the manual
	// reverse topo order backward.
	L.Backward()

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
