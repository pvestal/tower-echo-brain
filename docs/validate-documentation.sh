#!/bin/bash
# Documentation Validation Script for Echo Brain
# Validates the comprehensive documentation overhaul

echo "=================================================="
echo "Echo Brain Documentation Overhaul Validation"
echo "=================================================="
echo "Timestamp: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check file existence and size
check_file() {
    local file="$1"
    local description="$2"
    local min_size="$3"

    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ "$size" -gt "$min_size" ]; then
            echo -e "${GREEN}✓${NC} $description ($size bytes)"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} $description (exists but too small: $size bytes)"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} $description (missing)"
        return 1
    fi
}

# Function to validate YAML syntax
validate_yaml() {
    local file="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import yaml
try:
    with open('$file', 'r') as f:
        yaml.safe_load(f)
    print('✓ Valid YAML syntax')
except Exception as e:
    print(f'✗ Invalid YAML: {e}')
    exit(1)
" 2>/dev/null
        return $?
    else
        echo "⚠ Python3 not available for YAML validation"
        return 0
    fi
}

# Function to count content metrics
count_metrics() {
    local file="$1"
    local pattern="$2"
    local description="$3"

    if [ -f "$file" ]; then
        count=$(grep -c "$pattern" "$file" 2>/dev/null || echo "0")
        echo "   $description: $count"
    fi
}

DOCS_DIR="/opt/tower-echo-brain/docs"
cd "$DOCS_DIR" || exit 1

echo -e "${BLUE}1. Core Documentation Files${NC}"
echo "================================"

passed=0
total=0

# Check main documentation files
files=(
    "README.md:Documentation Hub (Main Index):5000"
    "openapi.yaml:OpenAPI Specification:20000"
    "swagger-ui.html:Interactive API Documentation:10000"
    "quick-start-guide.md:Quick Start Guide & Tutorials:30000"
    "user-journey-maps.md:User Journey Maps:25000"
    "tower-integration-patterns.md:Integration Patterns:25000"
    "troubleshooting-playbook.md:Troubleshooting Playbook:35000"
)

for file_info in "${files[@]}"; do
    IFS=':' read -r filename description min_size <<< "$file_info"
    if check_file "$filename" "$description" "$min_size"; then
        ((passed++))
    fi
    ((total++))
done

echo ""
echo -e "${BLUE}2. OpenAPI Specification Validation${NC}"
echo "===================================="

if [ -f "openapi.yaml" ]; then
    echo -n "YAML Syntax: "
    validate_yaml "openapi.yaml"

    echo "Content Analysis:"
    count_metrics "openapi.yaml" "^  /" "API Endpoints"
    count_metrics "openapi.yaml" "^    [a-z]*:" "HTTP Methods"
    count_metrics "openapi.yaml" "components:" "Component Schemas"
    count_metrics "openapi.yaml" "tags:" "API Tags"
else
    echo -e "${RED}✗ OpenAPI specification missing${NC}"
fi

echo ""
echo -e "${BLUE}3. Documentation Content Analysis${NC}"
echo "=================================="

echo "Quick Start Guide:"
count_metrics "quick-start-guide.md" "^###" "Tutorials"
count_metrics "quick-start-guide.md" "curl" "API Examples"
count_metrics "quick-start-guide.md" "```" "Code Blocks"

echo ""
echo "User Journey Maps:"
count_metrics "user-journey-maps.md" "## .*Persona" "User Personas"
count_metrics "user-journey-maps.md" "### Journey Map:" "Journey Maps"
count_metrics "user-journey-maps.md" "#### Phase" "Journey Phases"

echo ""
echo "Integration Patterns:"
count_metrics "tower-integration-patterns.md" "### .*Pattern" "Integration Patterns"
count_metrics "tower-integration-patterns.md" "```" "Code Examples"
count_metrics "tower-integration-patterns.md" "mermaid" "Architecture Diagrams"

echo ""
echo "Troubleshooting Playbook:"
count_metrics "troubleshooting-playbook.md" "### Issue" "Documented Issues"
count_metrics "troubleshooting-playbook.md" "```bash" "Diagnostic Commands"
count_metrics "troubleshooting-playbook.md" "**Solutions:**" "Solution Sections"

echo ""
echo -e "${BLUE}4. Interactive Features Validation${NC}"
echo "==================================="

# Check Swagger UI functionality
if [ -f "swagger-ui.html" ]; then
    echo -n "Swagger UI Setup: "
    if grep -q "SwaggerUIBundle" "swagger-ui.html"; then
        echo -e "${GREEN}✓${NC} SwaggerUI JavaScript included"
    else
        echo -e "${RED}✗${NC} SwaggerUI JavaScript missing"
    fi

    echo -n "Environment Switching: "
    if grep -q "switchEnvironment" "swagger-ui.html"; then
        echo -e "${GREEN}✓${NC} Multi-environment support"
    else
        echo -e "${YELLOW}⚠${NC} Environment switching missing"
    fi

    echo -n "Real-time Features: "
    if grep -q "WebSocket\|wscat" "swagger-ui.html"; then
        echo -e "${GREEN}✓${NC} WebSocket integration"
    else
        echo -e "${YELLOW}⚠${NC} WebSocket features missing"
    fi
fi

echo ""
echo -e "${BLUE}5. Documentation Server Validation${NC}"
echo "==================================="

if [ -f "serve-docs.py" ]; then
    echo -n "Documentation Server: "
    if [ -x "serve-docs.py" ]; then
        echo -e "${GREEN}✓${NC} Server script executable"
    else
        echo -e "${YELLOW}⚠${NC} Server script not executable"
    fi

    echo -n "Server Features: "
    features=0
    if grep -q "serve_swagger_ui" "serve-docs.py"; then ((features++)); fi
    if grep -q "serve_openapi_spec" "serve-docs.py"; then ((features++)); fi
    if grep -q "serve_health_check" "serve-docs.py"; then ((features++)); fi
    echo -e "${GREEN}✓${NC} $features/3 core features implemented"
fi

echo ""
echo -e "${BLUE}6. Documentation Coverage Assessment${NC}"
echo "====================================="

# Calculate documentation debt elimination score
total_files=7
files_complete=0
features_documented=0

echo "File Completeness:"
for file_info in "${files[@]}"; do
    IFS=':' read -r filename description min_size <<< "$file_info"
    if [ -f "$filename" ] && [ $(wc -c < "$filename") -gt "$min_size" ]; then
        ((files_complete++))
        echo -e "  ${GREEN}✓${NC} $description"
    else
        echo -e "  ${RED}✗${NC} $description"
    fi
done

echo ""
echo "Feature Documentation Coverage:"

# Check for key feature documentation
features=(
    "Model Management:models/manage"
    "Board of Directors:board/task"
    "Universal Testing:test/{target}"
    "Brain Visualization:brain"
    "Voice Integration:voice/notify"
    "WebSocket Streaming:stream"
    "Authentication:JWT"
    "Integration Patterns:Integration Patterns"
)

for feature_info in "${features[@]}"; do
    IFS=':' read -r feature_name search_term <<< "$feature_info"

    found=false
    for doc_file in *.md *.yaml; do
        if [ -f "$doc_file" ] && grep -q "$search_term" "$doc_file" 2>/dev/null; then
            found=true
            break
        fi
    done

    if $found; then
        echo -e "  ${GREEN}✓${NC} $feature_name"
        ((features_documented++))
    else
        echo -e "  ${RED}✗${NC} $feature_name"
    fi
done

echo ""
echo -e "${BLUE}7. Documentation Quality Metrics${NC}"
echo "================================="

# Calculate metrics
file_completeness=$((files_complete * 100 / total_files))
feature_coverage=$((features_documented * 100 / ${#features[@]}))
overall_score=$(((file_completeness + feature_coverage) / 2))

echo "Metrics Summary:"
echo "  File Completeness: $file_completeness% ($files_complete/$total_files files)"
echo "  Feature Coverage: $feature_coverage% ($features_documented/${#features[@]} features)"
echo "  Overall Quality Score: $overall_score%"

# Determine overall status
if [ "$overall_score" -ge 90 ]; then
    status_color="$GREEN"
    status="EXCELLENT"
elif [ "$overall_score" -ge 75 ]; then
    status_color="$YELLOW"
    status="GOOD"
elif [ "$overall_score" -ge 60 ]; then
    status_color="$YELLOW"
    status="ACCEPTABLE"
else
    status_color="$RED"
    status="NEEDS IMPROVEMENT"
fi

echo ""
echo "=================================================="
echo -e "Documentation Status: ${status_color}$status${NC} ($overall_score%)"
echo "=================================================="

echo ""
echo -e "${BLUE}8. Next Steps & Recommendations${NC}"
echo "==================================="

if [ "$overall_score" -ge 90 ]; then
    echo -e "${GREEN}✓${NC} Documentation overhaul successfully completed!"
    echo "  • Ready for production use"
    echo "  • All major components documented"
    echo "  • Interactive features working"
    echo ""
    echo "Recommended actions:"
    echo "  • Start documentation server: python3 serve-docs.py 8000"
    echo "  • Test interactive API docs: http://localhost:8000/api/"
    echo "  • Share quick start guide with team"
elif [ "$overall_score" -ge 75 ]; then
    echo -e "${YELLOW}⚠${NC} Documentation is good but has room for improvement"
    echo ""
    echo "Priority improvements:"
    if [ "$file_completeness" -lt 100 ]; then
        echo "  • Complete missing documentation files"
    fi
    if [ "$feature_coverage" -lt 100 ]; then
        echo "  • Document remaining features"
    fi
else
    echo -e "${RED}✗${NC} Documentation needs significant improvement"
    echo ""
    echo "Critical issues to address:"
    echo "  • Complete core documentation files"
    echo "  • Add missing feature documentation"
    echo "  • Validate content quality and accuracy"
fi

echo ""
echo "Testing Commands:"
echo "  # Start documentation server"
echo "  cd /opt/tower-echo-brain/docs && python3 serve-docs.py 8000"
echo ""
echo "  # Test API documentation"
echo "  curl http://localhost:8000/health"
echo ""
echo "  # Validate OpenAPI spec"
echo "  curl http://localhost:8000/openapi.yaml | head -20"

echo ""
echo "Documentation overhaul validation complete!"
echo "Timestamp: $(date)"

# Exit with appropriate code
if [ "$overall_score" -ge 75 ]; then
    exit 0
else
    exit 1
fi