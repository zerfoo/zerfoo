#!/bin/bash
# Zerfoo Documentation Generator Script

echo "🚀 Generating Zerfoo Documentation..."

# Ensure we're in the project root
if [ ! -f "go.mod" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Create directories
mkdir -p docs/api docs/generated docs/html scripts

echo "📦 Generating package documentation..."

# Generate Go documentation
go doc -all . > docs/api/root.txt 2>/dev/null || echo "⚠️  Root package documentation failed"

# Generate for each package
for pkg in tensor graph compute layers/core layers/activations training/optimizer training/loss distributed model pkg/tokenizer device; do
    if [ -d "$pkg" ] && ls "$pkg"/*.go 1> /dev/null 2>&1; then
        echo "  📄 Documenting $pkg..."
        pkg_name="${pkg//\//_}"
        go doc -all "./$pkg" > "docs/api/${pkg_name}.txt" 2>/dev/null || echo "⚠️  Failed to document $pkg"
    fi
done

echo "✅ Documentation generation complete!"
echo ""
echo "📖 View documentation:"
echo "  • Local server: godoc -http=:6060"
echo "  • Online: https://pkg.go.dev/github.com/zerfoo/zerfoo"
echo "  • Files: docs/api/*.txt"
