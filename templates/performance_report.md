# Python Performance Analysis Report

*Generated on {{ generated_at | format_datetime }}*

## Analysis Summary

{{ narrative }}

## Resource Metrics

- **Peak Memory Usage:** {{ analysis.metrics.peak_memory_mb | format_size }}
- **Average CPU Usage:** {{ analysis.metrics.avg_cpu_percent }}%
- **Execution Time:** {{ analysis.metrics.exec_time_ms }}ms

## Performance Issues

{% for issue in analysis.issues %}
### {{ issue.issue_type }}

**Location:** {{ issue.location }}  
**Priority Level:** {{ issue.priority_level | title }}

#### Original Code
```python
{{ issue.solution.original }}
```

#### Optimized Code
```python
{{ issue.solution.optimized }}
```

**Performance Impact:**  
{{ issue.solution.impact_summary }}

{% if issue.solution.reasoning %}
**Optimization Reasoning:**  
{{ issue.solution.reasoning }}
{% endif %}

---
{% endfor %}

## Hotspots Analysis

Top performance hotspots identified:

{% for hotspot in metrics.hotspots %}
- **{{ hotspot.function }}** (Line {{ hotspot.line }})
  - Hits: {{ hotspot.hits }}
  - Time: {{ hotspot.time_us }}μs
{% endfor %}

---
*End of Report*