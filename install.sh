#!/bin/bash
# NIS-TOOLKIT-SUIT Unified Installer
# The Official SDK for Organica AI's NIS Protocol Ecosystem
# Install both nis-devkit and nis-agentkit with one command

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "🚀 NIS-TOOLKIT-SUIT Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Official SDK for Organica AI's NIS Protocol Ecosystem"
echo "🔧 Installing: nis-core-toolkit + nis-agent-toolkit + nis-integrity-toolkit"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

# Check Python version
echo -e "${CYAN}🔍 Checking system requirements...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "  ✅ Python $python_version (>= 3.9 required)"
else
    echo -e "  ${RED}❌ Python $python_version detected. Python >= 3.9 required${NC}"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    echo -e "  ✅ pip3 available"
else
    echo -e "  ${RED}❌ pip3 not found. Please install pip3${NC}"
    exit 1
fi

# Check git
if command -v git &> /dev/null; then
    echo -e "  ✅ git available"
else
    echo -e "  ${RED}❌ git not found. Please install git${NC}"
    exit 1
fi

echo ""

# Installation options - Dual-Track Architecture
echo -e "${YELLOW}📋 Installation Options (Dual-Track Architecture):${NC}"
echo "  1) 🧠 Full Dual-Track Installation (recommended)"
echo "  2) 🔧 NDT Only - Human Developers (System Design)"
echo "  3) 🤖 NAT Only - AI Agents (Modular Cognition)"
echo "  4) 🛡️ Integrity Toolkit Only"
echo "  5) 👨‍💻 Development Installation (editable)"
echo ""
echo -e "${CYAN}💡 Dual-Track Info:${NC}"
echo "  NDT = NIS Developer Toolkit (for building multi-agent systems)"
echo "  NAT = NIS Agent Toolkit (for building agent minds)"
echo ""

read -p "Select installation type (1-5): " install_type

case $install_type in
    1)
        echo -e "${GREEN}🎯 Installing Full Dual-Track NIS-TOOLKIT-SUIT...${NC}"
        echo -e "${PURPLE}🧠 Both NDT (system design) and NAT (agent cognition)${NC}"
        install_full=true
        install_core=true
        install_agent=true
        install_integrity=true
        ;;
    2)
        echo -e "${GREEN}🎯 Installing NDT (NIS Developer Toolkit)...${NC}"
        echo -e "${BLUE}🔧 For human developers building multi-agent systems${NC}"
        install_full=false
        install_core=true
        install_agent=false
        install_integrity=true
        ;;
    3)
        echo -e "${GREEN}🎯 Installing NAT (NIS Agent Toolkit)...${NC}"
        echo -e "${GREEN}🤖 For AI agents with modular cognition${NC}"
        install_full=false
        install_core=false
        install_agent=true
        install_integrity=false
        ;;
    4)
        echo -e "${GREEN}🎯 Installing Integrity Toolkit Only...${NC}"
        echo -e "${RED}🛡️ Engineering integrity and quality assurance${NC}"
        install_full=false
        install_core=false
        install_agent=false
        install_integrity=true
        ;;
    5)
        echo -e "${GREEN}🎯 Installing Development Version (editable)...${NC}"
        echo -e "${YELLOW}👨‍💻 Full dual-track with development tools${NC}"
        install_full=true
        install_core=true
        install_agent=true
        install_integrity=true
        dev_install=true
        ;;
    *)
        echo -e "${RED}❌ Invalid selection${NC}"
        exit 1
        ;;
esac

echo ""

# Create virtual environment option
read -p "Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}🔧 Creating virtual environment...${NC}"
    python3 -m venv nis-toolkit-env
    source nis-toolkit-env/bin/activate
    echo -e "  ✅ Virtual environment created and activated"
    echo -e "  ${YELLOW}💡 Remember to run 'source nis-toolkit-env/bin/activate' in future sessions${NC}"
    echo ""
fi

# Install components
if [ "$install_core" = true ]; then
    echo -e "${PURPLE}📦 Installing NIS Core Toolkit...${NC}"
    if [ "$dev_install" = true ]; then
        pip3 install -e ./nis-core-toolkit/
    else
        pip3 install ./nis-core-toolkit/
    fi
    echo -e "  ✅ NIS Core Toolkit installed"
fi

if [ "$install_agent" = true ]; then
    echo -e "${PURPLE}📦 Installing NIS Agent Toolkit...${NC}"
    if [ "$dev_install" = true ]; then
        pip3 install -e ./nis-agent-toolkit/
    else
        pip3 install ./nis-agent-toolkit/
    fi
    echo -e "  ✅ NIS Agent Toolkit installed"
fi

if [ "$install_integrity" = true ]; then
    echo -e "${PURPLE}📦 Installing NIS Integrity Toolkit...${NC}"
    # For integrity toolkit, we'll just copy it to a standard location
    mkdir -p ~/.nis/integrity-toolkit
    cp -r ./nis-integrity-toolkit/* ~/.nis/integrity-toolkit/
    
    # Add to PATH if not already there
    if ! echo $PATH | grep -q "$HOME/.nis/integrity-toolkit"; then
        echo 'export PATH="$HOME/.nis/integrity-toolkit:$PATH"' >> ~/.bashrc
        export PATH="$HOME/.nis/integrity-toolkit:$PATH"
    fi
    echo -e "  ✅ NIS Integrity Toolkit installed"
fi

echo ""

# Verify installation
echo -e "${CYAN}🔍 Verifying installation...${NC}"

if [ "$install_core" = true ]; then
    if python3 -c "import nis_core_toolkit" 2>/dev/null; then
        echo -e "  ✅ NIS Core Toolkit import test passed"
    else
        echo -e "  ${YELLOW}⚠️  NIS Core Toolkit import test failed (may still work)${NC}"
    fi
fi

if [ "$install_agent" = true ]; then
    if python3 -c "import nis_agent_toolkit" 2>/dev/null; then
        echo -e "  ✅ NIS Agent Toolkit import test passed"
    else
        echo -e "  ${YELLOW}⚠️  NIS Agent Toolkit import test failed (may still work)${NC}"
    fi
fi

# Set up configuration directory
echo -e "${CYAN}🔧 Setting up configuration...${NC}"
mkdir -p ~/.nis
mkdir -p ~/.nis/projects
mkdir -p ~/.nis/templates
mkdir -p ~/.nis/logs

# Create example configuration
cat > ~/.nis/config.yaml << EOF
# NIS-TOOLKIT-SUIT Configuration
version: "1.0.0"
ecosystem:
  protocol_version: "3.0"
  compatible_systems:
    - NIS-HUB
    - NIS-X
    - NIS-DRONE
    - SparkNova
    - Orion
    - AlphaCortex
    - ArchaeologicalResearch

toolkit:
  core_toolkit_enabled: true
  agent_toolkit_enabled: true
  integrity_toolkit_enabled: true
  
development:
  auto_validation: true
  integrity_checks: true
  logging_level: "INFO"
  
projects:
  default_template: "advanced"
  auto_git_init: true
  include_examples: true
EOF

echo -e "  ✅ Configuration directory created at ~/.nis/"

echo ""

# Success banner
echo -e "${GREEN}"
echo "🎉 NIS-TOOLKIT-SUIT Dual-Track Installation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$install_core" = true ] && [ "$install_agent" = true ]; then
    echo "🧠 Dual-Track Commands Available:"
    echo ""
    echo "🔧 NDT (Human Developers - System Design):"
    echo "   nis init <project>          - Initialize new NIS project"
    echo "   nis create agent <name>     - Create intelligent agent"
    echo "   nis validate               - Validate project compliance"
    echo "   nis deploy                 - Deploy to platforms"
    echo "   nis connect                - Connect to NIS ecosystem"
    echo ""
    echo "🤖 NAT (AI Agents - Modular Cognition):"
    echo "   nis-agent create <name>     - Create agent with templates"
    echo "   nis-agent simulate         - Run agent simulations"
    echo "   nis-agent test             - Test agent functionality"
    echo ""
elif [ "$install_core" = true ]; then
    echo "🔧 NDT Commands (Human Developers):"
    echo "   nis init <project>          - Initialize new NIS project"
    echo "   nis create agent <name>     - Create intelligent agent"
    echo "   nis validate               - Validate project compliance"
    echo "   nis deploy                 - Deploy to platforms"
    echo "   nis connect                - Connect to NIS ecosystem"
    echo ""
elif [ "$install_agent" = true ]; then
    echo "🤖 NAT Commands (AI Agents):"
    echo "   nis-agent create <name>     - Create agent with templates"
    echo "   nis-agent simulate         - Run agent simulations"
    echo "   nis-agent test             - Test agent functionality"
    echo ""
fi

if [ "$install_integrity" = true ]; then
    echo "🛡️ Integrity Commands:"
    echo "   nis-integrity audit        - Run integrity checks"
    echo "   nis-integrity monitor      - Monitor project health"
    echo ""
fi

echo "📚 Documentation:"
echo "   README.md                   - Getting started guide"
echo "   docs/dual_track_architecture.md - Core design philosophy"
echo "   docs/compatibility_matrix.md - Protocol compatibility"
echo "   examples/                   - Working examples"
echo ""
echo "🌐 Ecosystem Integration:"
echo "   • Compatible with all NIS Protocol systems"
echo "   • Required dependency for NIS-HUB, NIS-X, NIS-DRONE"
echo "   • Integrates with SparkNova IDE"
echo ""

if [ "$install_core" = true ] && [ "$install_agent" = true ]; then
    echo "🚀 Dual-Track Quick Start:"
    echo "   # NDT: Build multi-agent system"
    echo "   nis init my-nis-project --template advanced"
    echo "   cd my-nis-project"
    echo "   nis create agent reasoning-agent --type reasoning"
    echo "   nis validate && nis deploy"
    echo ""
    echo "   # NAT: Build agent mind"
    echo "   nis-agent create cognitive-agent --template consciousness"
    echo "   nis-agent simulate --scenario complex-reasoning"
    echo ""
elif [ "$install_core" = true ]; then
    echo "🚀 NDT Quick Start:"
    echo "   nis init my-nis-project"
    echo "   cd my-nis-project"
    echo "   nis create agent reasoning-agent --type reasoning"
    echo "   nis validate && nis deploy"
    echo ""
elif [ "$install_agent" = true ]; then
    echo "🚀 NAT Quick Start:"
    echo "   nis-agent create cognitive-agent --template reasoning"
    echo "   nis-agent simulate --scenario complex-reasoning"
    echo "   nis-agent deploy --mode autonomous"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

# Create desktop shortcut option
read -p "Create desktop shortcut for NIS commands? [y/N]: " create_shortcut
if [[ $create_shortcut =~ ^[Yy]$ ]]; then
    if [ -d "$HOME/Desktop" ]; then
        cat > "$HOME/Desktop/NIS-Toolkit.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=NIS Toolkit
Comment=Organica AI NIS Protocol Development Toolkit
Exec=gnome-terminal -- bash -c "echo 'NIS-TOOLKIT-SUIT Commands:'; echo 'nis init <project> - Create project'; echo 'nis create agent <name> - Create agent'; echo 'nis validate - Validate project'; echo 'nis deploy - Deploy system'; echo ''; echo 'Type any command to get started:'; bash"
Icon=applications-development
Terminal=false
Categories=Development;
EOF
        chmod +x "$HOME/Desktop/NIS-Toolkit.desktop"
        echo -e "  ✅ Desktop shortcut created"
    fi
fi

echo ""
echo -e "${BLUE}💫 Welcome to the NIS Protocol Dual-Track Ecosystem!${NC}"
echo -e "${BLUE}   🔧 Build systems with NDT. 🤖 Build minds with NAT. 🧠 Build the future with both.${NC}" 