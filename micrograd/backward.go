package micrograd

import "slices"

// Backward propagates gradients in reverse topological order.
// Grads accumulates across calls, nothing resets them.
func (v *Value) Backward() {
	order := buildTopo(v, map[*Value]struct{}{}, []*Value{})
	slices.Reverse(order)

	order[0].Grad = 1.0

	// propagate gradients in reverse topo order
	for _, v := range order {
		v.propagate()
	}
}

// buildTopo performs DFS on the the graph rooted at cur to build the
// topological order of the operand graph.
func buildTopo(cur *Value, visited map[*Value]struct{}, order []*Value) []*Value {
	if cur == nil {
		return order
	}

	if _, ok := visited[cur]; ok {
		return order
	}

	visited[cur] = struct{}{}
	for _, n := range cur.prev {
		order = buildTopo(n, visited, order)
	}

	return append(order, cur)
}
