# NAT: NIS Agent Toolkit

**The Official Agent Development Framework for the NIS Protocol v3.1**

---

## 🚀 Key Features

-   **Unified Agent Framework:** A `BaseNISAgent` that implements the complete `Laplace → Consciousness → KAN → PINN → Safety` pipeline.
-   **`BitNet` Agent Template:** A pre-built agent template for creating agents with powerful, offline-first capabilities.
-   **Advanced Consciousness Integration:** A suite of tools for building self-aware AI systems with bias detection and ethical reasoning.
-   **Generative Simulation:** A framework for creating agents that can generate and test physically realistic 3D models.

---

##  Quick Start: Create a `BitNet` Agent

The NAT provides a simple and powerful CLI for creating new agents. Here's how to create a `BitNet`-powered agent for offline inference:

```bash
# Run from your NIS project directory
python ../nis-agent-toolkit/cli/main.py create bitnet my-offline-agent
```

This will create a new agent named `my-offline-agent` in your `agents/` directory, complete with a pre-configured `BitNet` model and the full unified pipeline.

---

## 🧠 Architecture: The `BaseNISAgent`

The `BaseNISAgent` is the heart of the NAT. It is an abstract base class that provides a complete, end-to-end implementation of the unified pipeline. All agents created with the NAT inherit from this class, giving them access to the full power of the NIS Protocol.

### Agent Types

| Type | Description | Status |
|---|---|---|
| **Reasoning** | Chain of Thought, problem solving | ✅ Working |
| **Vision** | Image processing, object detection | ✅ Template |
| **Memory** | Information storage, retrieval | ✅ Template |
| **Action** | Decision execution, command running | ✅ Template |
| **BitNet** | Offline-first, high-performance reasoning | ✅ Template |

---

*For more information on the overall architecture and the unified pipeline, please see the main project [README.md](../../README.md) and the [architecture documentation](../../docs/architecture.md).*
