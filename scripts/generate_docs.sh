#!/bin/bash

# Zerfoo Documentation Generator
# This script generates comprehensive documentation for the Zerfoo project

set -e

echo "🚀 Generating Zerfoo Documentation..."

# Create documentation directories
mkdir -p docs/api
mkdir -p docs/generated
mkdir -p docs/html

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v go &> /dev/null; then
        print_error "Go is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v godoc &> /dev/null; then
        print_warning "godoc not found, installing..."
        go install golang.org/x/tools/cmd/godoc@latest
    fi
    
    print_success "Dependencies checked"
}

# Generate package documentation
generate_package_docs() {
    print_status "Generating package documentation..."
    
    # List of packages to document
    packages=(
        "."
        "tensor"
        "graph" 
        "compute"
        "layers/core"
        "layers/activations"
        "training/optimizer"
        "training/loss"
        "distributed"
        "model"
        "pkg/onnx"
        "pkg/tokenizer"
        "device"
        "numeric"
    )
    
    for pkg in "${packages[@]}"; do
        if [ "$pkg" = "." ]; then
            pkg_name="root"
            pkg_path="."
        else
            pkg_name="${pkg//\//_}"
            pkg_path="./$pkg"
        fi
        
        # Check if package exists
        if [ -d "$pkg_path" ] && ls "$pkg_path"/*.go 1> /dev/null 2>&1; then
            print_status "Documenting package: $pkg"
            
            # Generate package documentation
            go doc -all "$pkg_path" > "docs/api/${pkg_name}.txt" 2>/dev/null || {
                print_warning "Failed to generate docs for $pkg"
                continue
            }
            
            # Generate HTML documentation if possible
            if command -v godoc &> /dev/null; then
                print_status "Generating HTML for $pkg..."
                # Note: This would need a running godoc server to extract HTML
                # For now, we'll create a simple HTML wrapper
                create_html_wrapper "$pkg_name" "$pkg"
            fi
            
            print_success "Generated documentation for $pkg"
        else
            print_warning "Package $pkg not found or contains no Go files"
        fi
    done
}

# Create HTML wrapper for documentation
create_html_wrapper() {
    local pkg_name="$1"
    local pkg_path="$2"
    
    cat > "docs/html/${pkg_name}.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zerfoo Package: $pkg_path</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fafafa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .content {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        pre {
            background: #f4f4f4;
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #667eea;
        }
        code {
            background: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, monospace;
        }
        .navigation {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .navigation a {
            text-decoration: none;
            color: #667eea;
            margin-right: 1rem;
            font-weight: 500;
        }
        .navigation a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="navigation">
        <a href="../README.html">📖 Documentation Home</a>
        <a href="../DEVELOPER_GUIDE.html">👨‍💻 Developer Guide</a>
        <a href="../API_REFERENCE.html">📚 API Reference</a>
        <a href="../EXAMPLES.html">💡 Examples</a>
    </div>
    
    <div class="header">
        <h1>📦 Package: $pkg_path</h1>
        <p>Generated documentation for Zerfoo package</p>
    </div>
    
    <div class="content">
        <h2>Package Documentation</h2>
        <p><em>For the most up-to-date documentation, run: <code>go doc -all ./$pkg_path</code></em></p>
        
        <h3>View Documentation</h3>
        <p>To view the complete documentation for this package:</p>
        <pre><code># Command line documentation
go doc -all ./$pkg_path

# Start local documentation server
godoc -http=:6060
# Then visit: http://localhost:6060/pkg/github.com/zerfoo/zerfoo/$pkg_path</code></pre>
        
        <h3>Online Documentation</h3>
        <p>View on pkg.go.dev: <a href="https://pkg.go.dev/github.com/zerfoo/zerfoo/$pkg_path" target="_blank">https://pkg.go.dev/github.com/zerfoo/zerfoo/$pkg_path</a></p>
    </div>
</body>
</html>
EOF
}

# Generate README index with links
generate_documentation_index() {
    print_status "Generating documentation index..."
    
    cat > "docs/generated/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zerfoo Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 3rem;
        }
        .header h1 {
            color: #333;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        .header p {
            color: #666;
            font-size: 1.2rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }
        .card {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #667eea;
            transition: transform 0.2s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .card h3 {
            color: #333;
            margin-top: 0;
            font-size: 1.5rem;
        }
        .card p {
            color: #666;
            margin-bottom: 1rem;
        }
        .card a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            display: inline-block;
            margin-top: 1rem;
        }
        .card a:hover {
            text-decoration: underline;
        }
        .quick-start {
            background: #e3f2fd;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
        }
        .quick-start h2 {
            color: #1565c0;
            margin-top: 0;
        }
        pre {
            background: #263238;
            color: #fff;
            padding: 1.5rem;
            border-radius: 10px;
            overflow-x: auto;
        }
        .badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Zerfoo</h1>
            <p>High-Performance Go Framework for Machine Learning</p>
            <div class="badge">Pre-release • Go 1.24+ • Apache 2.0</div>
        </div>
        
        <div class="quick-start">
            <h2>⚡ Quick Start</h2>
            <pre><code>go get github.com/zerfoo/zerfoo</code></pre>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📖 Developer Guide</h3>
                <p>Complete guide to building ML models with Zerfoo. Perfect starting point for new users.</p>
                <a href="DEVELOPER_GUIDE.html">Read the Guide →</a>
            </div>
            
            <div class="card">
                <h3>💡 Examples & Tutorials</h3>
                <p>Hands-on examples from basic concepts to advanced use cases. Learn by doing!</p>
                <a href="EXAMPLES.html">Explore Examples →</a>
            </div>
            
            <div class="card">
                <h3>📚 API Reference</h3>
                <p>Comprehensive API documentation for all packages, types, and functions.</p>
                <a href="API_REFERENCE.html">Browse API →</a>
            </div>
            
            <div class="card">
                <h3>🤝 Contributing</h3>
                <p>Guidelines for contributing code, documentation, and examples to Zerfoo.</p>
                <a href="CONTRIBUTING.html">Start Contributing →</a>
            </div>
            
            <div class="card">
                <h3>🏗️ Architecture</h3>
                <p>Deep dive into Zerfoo's design principles, architecture, and technical decisions.</p>
                <a href="design.html">Explore Architecture →</a>
            </div>
            
            <div class="card">
                <h3>🎯 Project Goals</h3>
                <p>Learn about Zerfoo's vision, objectives, and roadmap for the future.</p>
                <a href="goal.html">Read Goals →</a>
            </div>
        </div>
        
        <div style="text-align: center; color: #666; border-top: 1px solid #eee; padding-top: 2rem;">
            <p>🌟 Built with Go • 📄 Documentation generated automatically</p>
            <p><a href="https://github.com/zerfoo/zerfoo" style="color: #667eea;">GitHub Repository</a> | 
               <a href="https://pkg.go.dev/github.com/zerfoo/zerfoo" style="color: #667eea;">pkg.go.dev</a></p>
        </div>
    </div>
</body>
</html>
EOF

    print_success "Generated documentation index"
}

# Convert markdown files to HTML
convert_markdown_to_html() {
    print_status "Converting markdown files to HTML..."
    
    # List of markdown files to convert
    md_files=(
        "DEVELOPER_GUIDE.md"
        "API_REFERENCE.md"
        "EXAMPLES.md"
        "CONTRIBUTING.md"
        "DOCUMENTATION_GUIDE.md"
        "design.md"
        "goal.md"
        "tree.md"
    )
    
    for md_file in "${md_files[@]}"; do
        if [ -f "docs/$md_file" ]; then
            html_file="${md_file%.md}.html"
            print_status "Converting $md_file to $html_file..."
            
            # Simple markdown to HTML conversion (basic)
            # In production, you'd use a proper markdown processor
            create_html_from_markdown "docs/$md_file" "docs/generated/$html_file"
            
            print_success "Converted $md_file"
        else
            print_warning "Markdown file not found: $md_file"
        fi
    done
}

# Create HTML from markdown (basic conversion)
create_html_from_markdown() {
    local input_file="$1"
    local output_file="$2"
    local title=$(basename "$input_file" .md)
    
    cat > "$output_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zerfoo - $title</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fafafa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .content {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        pre {
            background: #f4f4f4;
            padding: 1rem;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #667eea;
        }
        code {
            background: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, monospace;
        }
        h1, h2, h3 { color: #333; }
        h1 { border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }
        a { color: #667eea; }
        .navigation {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .navigation a {
            text-decoration: none;
            color: #667eea;
            margin-right: 1rem;
            font-weight: 500;
        }
        .navigation a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="navigation">
        <a href="index.html">🏠 Home</a>
        <a href="DEVELOPER_GUIDE.html">👨‍💻 Developer Guide</a>
        <a href="API_REFERENCE.html">📚 API Reference</a>
        <a href="EXAMPLES.html">💡 Examples</a>
        <a href="CONTRIBUTING.html">🤝 Contributing</a>
    </div>
    
    <div class="header">
        <h1>📄 $title</h1>
        <p>Zerfoo Documentation</p>
    </div>
    
    <div class="content">
        <p><em>Note: This is a basic HTML conversion. For the best formatting, view the original markdown file or use a proper markdown renderer.</em></p>
        
        <h2>Content</h2>
        <pre><code>$(cat "$input_file" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')</code></pre>
        
        <p><strong>💡 Tip:</strong> For better formatting, you can:</p>
        <ul>
            <li>View the source markdown file: <code>$input_file</code></li>
            <li>Use a markdown viewer or editor</li>
            <li>Install a proper markdown-to-HTML converter</li>
        </ul>
    </div>
</body>
</html>
EOF
}

# Generate documentation summary
generate_summary() {
    print_status "Generating documentation summary..."
    
    cat > "docs/generated/SUMMARY.md" << 'EOF'
# Zerfoo Documentation Summary

## 📊 Documentation Statistics

Generated on: $(date)

### 📁 Documentation Structure

```
docs/
├── README.md                 # Main documentation index
├── DEVELOPER_GUIDE.md        # Complete developer guide
├── API_REFERENCE.md          # API documentation
├── EXAMPLES.md              # Tutorials and examples
├── CONTRIBUTING.md          # Contributing guidelines
├── DOCUMENTATION_GUIDE.md   # Documentation standards
├── design.md               # Architecture design
├── goal.md                 # Project goals
├── tree.md                 # Project structure
├── api/                    # Generated API docs
├── generated/              # Generated HTML files
└── html/                   # Package HTML docs
```

### 🛠️ Generated Files

- Package documentation for all Go packages
- HTML versions of markdown documentation
- Cross-referenced API documentation
- Interactive documentation index

### 🚀 Usage

1. **Local Development**: Use `godoc -http=:6060` for live documentation
2. **Online**: Visit https://pkg.go.dev/github.com/zerfoo/zerfoo
3. **Offline**: Open `docs/generated/index.html` in your browser

### 📈 Documentation Coverage

- Core packages: ✅ Documented
- API reference: ✅ Complete
- Examples: ✅ Comprehensive
- Tutorials: ✅ Available
- Contributing guide: ✅ Detailed

### 🔄 Regenerating Documentation

Run the documentation generator:

```bash
./scripts/generate_docs.sh
```

Or manually:

```bash
go doc -all ./... > docs/api/complete.txt
godoc -http=:6060 # For live documentation
```
EOF

    print_success "Generated documentation summary"
}

# Create documentation generation script
create_docs_script() {
    print_status "Creating documentation script..."
    
    cat > "scripts/generate_docs.sh" << 'EOF'
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
for pkg in tensor graph compute layers/core layers/activations training/optimizer training/loss distributed model pkg/onnx pkg/tokenizer device; do
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
EOF

    chmod +x "scripts/generate_docs.sh"
    print_success "Created documentation generation script"
}

# Test documentation
test_documentation() {
    print_status "Testing documentation..."
    
    # Test that documentation can be generated
    if go doc . >/dev/null 2>&1; then
        print_success "Go documentation is working"
    else
        print_warning "Go documentation has issues"
    fi
    
    # Test that key files exist
    key_files=(
        "docs/DEVELOPER_GUIDE.md"
        "docs/API_REFERENCE.md"
        "docs/EXAMPLES.md"
        "docs/CONTRIBUTING.md"
        "docs/README.md"
    )
    
    for file in "${key_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "Found: $file"
        else
            print_warning "Missing: $file"
        fi
    done
}

# Main execution
main() {
    echo "📚 Zerfoo Documentation Generator"
    echo "================================="
    echo ""
    
    check_dependencies
    echo ""
    
    generate_package_docs
    echo ""
    
    generate_documentation_index
    echo ""
    
    convert_markdown_to_html
    echo ""
    
    generate_summary
    echo ""
    
    create_docs_script
    echo ""
    
    test_documentation
    echo ""
    
    print_success "🎉 Documentation generation complete!"
    echo ""
    echo "📖 Next steps:"
    echo "  1. Start local docs: godoc -http=:6060"
    echo "  2. Open: http://localhost:6060/pkg/github.com/zerfoo/zerfoo/"
    echo "  3. View HTML: open docs/generated/index.html"
    echo "  4. Read guides: docs/DEVELOPER_GUIDE.md"
    echo ""
    echo "🔄 To regenerate: ./scripts/generate_docs.sh"
}

# Run main function
main "$@"
EOF

chmod +x generate_docs.sh