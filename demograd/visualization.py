# This function is heavily inspired by the visualizations used by Karphathy in his Micrograd engine.

from graphviz import Digraph


def build_graphviz(tensor):
    """
    Recursively builds a computation graph using Graphviz's Digraph,
    starting from the given tensor.
    Each node represents either a Tensor (displaying its name or a generic label)
    or a Function (displaying its class name).
    """
    dot = Digraph(comment="Computation Graph")
    visited = set()

    def add_nodes(t):
        # Check if the tensor has been processed already
        if id(t) in visited:
            return
        visited.add(id(t))
        # Label the tensor node; use its name if available, otherwise show its shape.
        label = t.name if t.name else f"Tensor\nshape: {t.data.shape}"
        dot.node(str(id(t)), label, shape="box", style="filled", color="lightblue")

        # If this tensor was produced by a function, add that function node and connect it.
        if t._grad_fn is not None:
            fn = t._grad_fn
            fn_label = fn.__class__.__name__
            dot.node(
                str(id(fn)),
                fn_label,
                shape="ellipse",
                style="filled",
                color="lightgrey",
            )
            # Connect function to tensor (function produced tensor)
            dot.edge(str(id(fn)), str(id(t)))
            # For each input tensor, recursively add nodes and connect input to function.
            for inp in fn.inputs:
                add_nodes(inp)
                dot.edge(str(id(inp)), str(id(fn)))

    add_nodes(tensor)
    return dot


def visualize_graph(tensor, filename="computation_graph", format="pdf"):
    """
    Visualizes the computation graph starting from the given tensor.
    The graph is saved to a file (default: PDF format).
    """
    dot = build_graphviz(tensor)
    # Render and view the graph (this will create files like computation_graph.pdf)
    dot.render(filename, format=format, view=True)
