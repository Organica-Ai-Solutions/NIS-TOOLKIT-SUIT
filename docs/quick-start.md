# NIS Toolkit Suite - Quick Start Guide

Welcome to the **NIS Toolkit Suite**, the official development environment for the NIS Protocol. This guide will walk you through setting up your environment and creating your first NIS implementation from scratch.

---

## 1. Installation

First, clone the `NIS-TOOLKIT-SUIT` repository and install the required dependencies.

```bash
# Clone the main toolkit repository
git clone https://github.com/Organica-Ai-Solutions/NIS-TOOLKIT-SUIT.git
cd NIS-TOOLKIT-SUIT

# Install all required dependencies
pip install -r requirements.txt
```

This installs the NIS Developer Toolkit (NDT), NIS Agent Toolkit (NAT), and NIS Integrity Toolkit (NIT), giving you everything you need to build, test, and validate your projects.

---

## 2. Create Your First NIS Project

The NIS Developer Toolkit (`nis-core-toolkit`) provides a powerful `init` command to scaffold a new project for you. This is the recommended way to start any new NIS implementation.

Run the following command from the root of the `NIS-TOOLKIT-SUIT` directory:

```bash
# Usage: python nis-core-toolkit/cli/main.py init <your-project-name>
python nis-core-toolkit/cli/main.py init my-first-nis-project
```

This command creates a new directory (`my-first-nis-project`) with a complete, best-practices structure for a NIS Protocol project.

### What's Inside Your New Project?

The `init` command generates the following structure:

```
my-first-nis-project/
├── .gitignore
├── README.md
├── requirements.txt
├── agents/
│   └── .gitkeep
├── config/
│   ├── agent_config.yml
│   └── system_config.yml
├── data/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── scripts/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── cognitive_agents/
    ├── core/
    ├── llm/
    ├── memory/
    └── neural_hierarchy/
```

-   **`agents/`**: Your custom agents, created using the Agent Toolkit (NAT), will live here.
-   **`config/`**: Configuration files for your system and agents.
-   **`data/`**: A place for local data, datasets, and models.
-   **`notebooks/`**: Jupyter notebooks for exploration and analysis.
-   **`scripts/`**: Utility and automation scripts.
-   **`src/`**: The core source code of your NIS implementation, pre-populated with the base logic from the official NIS Protocol.

---

## 3. Develop Your First Agent

Now that your project is set up, you can start building. Use the NIS Agent Toolkit (NAT) to create your first agent within the new project.

```bash
# Navigate into your new project directory
cd my-first-nis-project

# Use the NAT to create a new "reasoning" agent
# Note: We call the script from its location within the NIS-TOOLKIT-SUIT
python ../nis-agent-toolkit/cli/main.py create reasoning my-reasoning-agent --consciousness-level 0.9 --kan-enabled
```

This will create a new agent named `my-reasoning-agent` inside the `agents/` directory of your project, complete with high consciousness and KAN mathematical reasoning capabilities.

---

## 4. Next Steps

Your NIS project is now set up and contains its first agent. From here, you can:

-   **Develop Your Agent's Logic:** Edit the files in `agents/my-reasoning-agent/` to implement its unique behavior.
-   **Build the Core System:** Add your custom logic to the `src/` directory to orchestrate your agents.
-   **Test Your Agent:** Use `nis-agent-toolkit` to run tests:
    ```bash
    python ../nis-agent-toolkit/cli/main.py test my-reasoning-agent
    ```
-   **Audit for Integrity:** Use the `nis-integrity-toolkit` to validate your project's quality and honesty:
    ```bash
    python ../nis-integrity-toolkit/cli/main.py audit full --project-path .
    ```

Congratulations! You have successfully created and started developing a new NIS implementation using the official toolkit. 