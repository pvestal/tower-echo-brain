#!/usr/bin/env python3
"""Generate dependency visualization for Echo Brain architecture"""

import os
import re
from pathlib import Path
from collections import defaultdict

def analyze_imports(src_dir="/opt/tower-echo-brain/src"):
    """Analyze all Python imports in the src directory"""

    imports = defaultdict(list)
    file_count = defaultdict(int)

    for root, dirs, files in os.walk(src_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, src_dir)
                dir_name = relative_path.split('/')[0] if '/' in relative_path else 'root'
                file_count[dir_name] += 1

                with open(filepath, 'r') as f:
                    content = f.read()

                    # Find all imports from src
                    import_pattern = r'from src\.(\w+)'
                    matches = re.findall(import_pattern, content)

                    for match in matches:
                        if match != dir_name:  # Don't count self-imports
                            imports[dir_name].append(match)

    return imports, file_count

def generate_mermaid_diagram(imports, file_count):
    """Generate Mermaid diagram for visualization"""

    diagram = ["```mermaid", "graph TD"]

    # Add nodes with file counts
    for dir_name, count in file_count.items():
        if dir_name != 'unused':  # Skip unused directory
            diagram.append(f"    {dir_name}[{dir_name}<br/>{count} files]")

    # Add edges with import counts
    edge_counts = defaultdict(int)
    for source, targets in imports.items():
        if source != 'unused':
            for target in targets:
                if target != 'unused':
                    edge_counts[(source, target)] += 1

    for (source, target), count in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True):
        diagram.append(f"    {source} -->|{count}| {target}")

    diagram.append("```")
    return "\n".join(diagram)

def generate_report():
    """Generate complete dependency analysis report"""

    imports, file_count = analyze_imports()

    report = ["# Echo Brain Dependency Analysis", ""]
    report.append("## Directory Statistics")
    report.append("| Directory | Files | Imported By | Total Imports |")
    report.append("|-----------|-------|-------------|---------------|")

    # Calculate statistics
    imported_by = defaultdict(set)
    total_imports = defaultdict(int)

    for source, targets in imports.items():
        for target in targets:
            imported_by[target].add(source)
            total_imports[target] += 1

    for dir_name in sorted(file_count.keys()):
        if dir_name != 'unused':
            files = file_count[dir_name]
            importers = len(imported_by[dir_name])
            total = total_imports[dir_name]
            report.append(f"| {dir_name} | {files} | {importers} | {total} |")

    report.append("")
    report.append("## Dependency Graph")
    report.append(generate_mermaid_diagram(imports, file_count))

    report.append("")
    report.append("## Key Insights")

    # Most imported
    most_imported = sorted(total_imports.items(), key=lambda x: x[1], reverse=True)[:5]
    report.append("\n### Most Imported Directories")
    for dir_name, count in most_imported:
        importers = ", ".join(sorted(imported_by[dir_name]))
        report.append(f"- **{dir_name}** ({count} imports) from: {importers}")

    # Least imported (candidates for consolidation)
    least_imported = [(k, total_imports[k]) for k in file_count.keys() if k != 'unused' and total_imports[k] <= 1]
    if least_imported:
        report.append("\n### Consolidation Candidates (≤1 import)")
        for dir_name, count in least_imported:
            report.append(f"- **{dir_name}** ({count} imports)")

    # Circular dependencies
    report.append("\n### Circular Dependencies")
    circular = []
    for source, targets in imports.items():
        for target in targets:
            if source in imports.get(target, []):
                pair = tuple(sorted([source, target]))
                if pair not in circular:
                    circular.append(pair)
                    report.append(f"- {pair[0]} ↔ {pair[1]}")

    if not circular:
        report.append("- None detected")

    return "\n".join(report)

if __name__ == "__main__":
    report = generate_report()
    output_file = "/opt/tower-echo-brain/DEPENDENCY_ANALYSIS.md"

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Dependency analysis saved to {output_file}")
    print("\nSummary:")
    print(report.split("## Key Insights")[1])