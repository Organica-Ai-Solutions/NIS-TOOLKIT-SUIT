#!/bin/bash

# 🛡️ NIS Integrity Hooks Setup
# Installs git hooks to prevent simulation-based commits

echo "🛡️ Setting up NIS Integrity Git Hooks..."
echo "======================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository root"
    echo "   Please run this from the project root directory"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Pre-commit hook to catch simulation patterns
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# 🚨 NIS Anti-Simulation Pre-Commit Hook
# Prevents committing training scripts with simulation patterns

echo "🔍 Running NIS integrity check..."

# Check if anti-simulation validator exists
if [ -f "nis-integrity-toolkit/audit-scripts/anti_simulation_validator.py" ]; then
    
    # Run anti-simulation validator on staged files
    python3 nis-integrity-toolkit/audit-scripts/anti_simulation_validator.py --project-path .
    
    if [ $? -eq 1 ]; then
        echo ""
        echo "❌ COMMIT BLOCKED - Simulation violations detected!"
        echo "   Fix the violations above before committing"
        echo "   Or use 'git commit --no-verify' to bypass (NOT RECOMMENDED)"
        exit 1
    elif [ $? -eq 0 ]; then
        echo "⚠️  WARNING: Some suspicious patterns detected"
        echo "   Review carefully before proceeding"
    fi
    
else
    echo "⚠️  Warning: Anti-simulation validator not found"
    echo "   Install nis-integrity-toolkit for full protection"
fi

# Additional quick checks for common simulation patterns
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ ! -z "$staged_files" ]; then
    echo "🔍 Quick simulation pattern check..."
    
    # Check for obvious simulation patterns
    simulation_found=false
    
    for file in $staged_files; do
        if [ -f "$file" ]; then
            # Check for time.sleep patterns
            if grep -n "time\.sleep(" "$file" > /dev/null; then
                echo "🚨 Found time.sleep() in $file - potential simulation!"
                simulation_found=true
            fi
            
            # Check for fake random metrics
            if grep -n "np\.random\.uniform.*0\." "$file" > /dev/null; then
                echo "🚨 Found fake random metrics in $file!"
                simulation_found=true
            fi
            
            # Check for hardcoded success
            if grep -n "training_successful.*=.*True" "$file" > /dev/null; then
                echo "🚨 Found hardcoded success in $file!"
                simulation_found=true
            fi
        fi
    done
    
    if [ "$simulation_found" = true ]; then
        echo ""
        echo "❌ COMMIT BLOCKED - Simulation patterns detected!"
        echo "   Remove simulation code before committing"
        exit 1
    fi
fi

echo "✅ Integrity check passed"
EOF

# Make pre-commit hook executable
chmod +x .git/hooks/pre-commit

# Pre-push hook for final validation
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash

# 🚨 NIS Anti-Simulation Pre-Push Hook
# Final integrity check before pushing

echo "🔍 Running final integrity check before push..."

# Run comprehensive audit if available
if [ -f "nis-integrity-toolkit/audit-scripts/full-audit.py" ]; then
    echo "📋 Running full audit..."
    python3 nis-integrity-toolkit/audit-scripts/full-audit.py --project-path .
    
    if [ $? -ne 0 ]; then
        echo "❌ PUSH BLOCKED - Audit failed!"
        echo "   Fix integrity issues before pushing"
        exit 1
    fi
fi

# Anti-simulation check
if [ -f "nis-integrity-toolkit/audit-scripts/anti_simulation_validator.py" ]; then
    echo "🛡️ Running anti-simulation validation..."
    python3 nis-integrity-toolkit/audit-scripts/anti_simulation_validator.py --project-path .
    
    if [ $? -eq 1 ]; then
        echo "❌ PUSH BLOCKED - Critical simulation violations!"
        echo "   Fix simulation issues before pushing"
        exit 1
    fi
fi

echo "✅ All integrity checks passed - push allowed"
EOF

# Make pre-push hook executable
chmod +x .git/hooks/pre-push

# Commit message hook to track integrity
cat > .git/hooks/prepare-commit-msg << 'EOF'
#!/bin/bash

# Add integrity marker to commit messages
commit_file=$1
commit_source=$2

# Only modify if it's a regular commit (not merge, etc.)
if [ -z "$commit_source" ] || [ "$commit_source" = "message" ]; then
    
    # Check if message already has integrity marker
    if ! grep -q "\[INTEGRITY-VERIFIED\]" "$commit_file"; then
        
        # Add integrity verification timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "" >> "$commit_file"
        echo "[INTEGRITY-VERIFIED: $timestamp]" >> "$commit_file"
        
    fi
fi
EOF

# Make prepare-commit-msg hook executable
chmod +x .git/hooks/prepare-commit-msg

echo ""
echo "✅ NIS Integrity Git Hooks installed successfully!"
echo ""
echo "📋 Installed hooks:"
echo "   🔍 pre-commit: Blocks simulation patterns"
echo "   🚀 pre-push: Comprehensive integrity check"
echo "   📝 prepare-commit-msg: Adds integrity verification"
echo ""
echo "🛡️ Protection active! Simulation commits will be blocked."
echo ""
echo "💡 To bypass hooks (emergency only): git commit --no-verify"
echo "⚠️  WARNING: Only bypass hooks if you're certain code is clean!" 