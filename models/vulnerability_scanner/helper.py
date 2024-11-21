from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from reportlab.platypus import Flowable
from reportlab.lib import colors


HTML_TEMPLATE = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Security Scan Report</title>
            <style>
                :root {
                    --bg-primary: #1a1a1a;
                    --bg-secondary: #2b2b2b;
                    --text-primary: #f0f0f0;
                    --text-secondary: #b0b0b0;
                    --accent-blue: #5da3fa;
                    --severity-high: #e74c3c;
                    --severity-medium: #f39c12;
                    --severity-low: #2ecc71;
                    --border-color: #444;
                }
                
                body {
                    background-color: var(--bg-primary);
                    color: var(--text-primary);
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                h1, h2, h3 {
                    color: var(--accent-blue);
                    margin-top: 30px;
                }
                
                .summary-cards {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                
                .card {
                    background-color: var(--bg-secondary);
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                
                .card h3 {
                    margin-top: 0;
                    font-size: 1.2em;
                    color: var(--text-primary);
                }
                
                .table-container {
                    background-color: var(--bg-secondary);
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    overflow-x: auto;
                }
                
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                
                th, td {
                    border: 1px solid var(--border-color);
                    padding: 12px;
                    text-align: left;
                }
                
                th {
                    background-color: var(--bg-primary);
                    color: var(--accent-blue);
                }
                
                tr:nth-child(even) {
                    background-color: rgba(255, 255, 255, 0.05);
                }
                
                .severity-high { color: var(--severity-high); }
                .severity-medium { color: var(--severity-medium); }
                .severity-low { color: var(--severity-low); }
                
                .vulnerability-details {
                    background-color: var(--bg-secondary);
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }
                
                .code-block {
                    background-color: var(--bg-primary);
                    border-radius: 4px;
                    padding: 15px;
                    margin: 10px 0;
                    overflow-x: auto;
                    font-family: 'Consolas', monospace;
                }
                
                .chart {
                    height: 300px;
                    margin: 20px 0;
                }
                
                @media (max-width: 768px) {
                    .summary-cards {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Security Scan Report</h1>
                <p>Generated on: {{ timestamp }}</p>
                
                <div class="summary-cards">
                    <div class="card">
                        <h3>Scan Overview</h3>
                        <p>Total Vulnerabilities: {{ summary.total_vulnerabilities }}</p>
                        <p>Files Scanned: {{ summary.files_affected }}</p>
                        <p>Lines Scanned: {{ summary.total_lines_scanned }}</p>
                    </div>
                    
                    <div class="card">
                        <h3>Risk Analysis</h3>
                        <p>Average Risk Score: {{ "%.2f"|format(summary.average_risk_score) }}</p>
                        <p>High Severity: {{ summary.severity_distribution.High }}</p>
                        <p>Medium Severity: {{ summary.severity_distribution.Medium }}</p>
                        <p>Low Severity: {{ summary.severity_distribution.Low }}</p>
                    </div>
                </div>

                <div class="table-container">
                    <h2>Vulnerability Summary</h2>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Count</th>
                            <th>Risk Level</th>
                        </tr>
                        {% for type, count in summary.vulnerability_types.items() %}
                        <tr>
                            <td>{{ type }}</td>
                            <td>{{ count }}</td>
                            <td class="severity-{{ 'high' if count > 2 else 'medium' if count > 1 else 'low' }}">
                                {{ 'High' if count > 2 else 'Medium' if count > 1 else 'Low' }}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>

                <div class="table-container">
                    <h2>Detailed Findings</h2>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>File</th>
                            <th>Line</th>
                            <th>Severity</th>
                            <th>Risk Score</th>
                            <th>CWE ID</th>
                        </tr>
                        {% for vuln in vulnerabilities %}
                        <tr>
                            <td>{{ vuln.vulnerability_type }}</td>
                            <td>{{ vuln.file_path or 'N/A' }}</td>
                            <td>{{ vuln.line_number }}</td>
                            <td class="severity-{{ vuln.severity|lower }}">{{ vuln.severity }}</td>
                            <td>{{ "%.1f"|format(vuln.risk_score) }}</td>
                            <td>{{ vuln.cwe_id or 'N/A' }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>

                <h2>Detailed Vulnerability Analysis</h2>
                {% for vuln in vulnerabilities %}
                <div class="vulnerability-details">
                    <h3>{{ vuln.vulnerability_type }}</h3>
                    <p><strong>Location:</strong> {{ vuln.file_path or 'N/A' }}:{{ vuln.line_number }}</p>
                    <p><strong>Severity:</strong> <span class="severity-{{ vuln.severity|lower }}">{{ vuln.severity }}</span></p>
                    <p><strong>Risk Score:</strong> {{ "%.1f"|format(vuln.risk_score) }}</p>
                    <p><strong>Description:</strong> {{ vuln.description }}</p>
                    <p><strong>Impact:</strong> {{ vuln.impact }}</p>
                    
                    <div class="code-block">
                        <p><strong>Vulnerable Code:</strong></p>
                        <pre>{{ vuln.code_snippet }}</pre>
                    </div>
                    
                    <div class="code-block">
                        <p><strong>Corrected Code:</strong></p>
                        <pre>{{ vuln.corrected_code }}</pre>
                    </div>
                    
                    <p><strong>Recommendation:</strong> {{ vuln.recommendation }}</p>
                    {% if vuln.cwe_id %}
                    <p><strong>CWE ID:</strong> {{ vuln.cwe_id }}</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """


class VulnerabilityType(str, Enum):
    """
    Enumeration of OWASP Top 10 vulnerability types.

    Provides predefined constants for OWASP 2021 vulnerability categories
    to standardize vulnerability detection and reporting.
    """

    INJECTION = "A01:2021-Injection"
    CRYPTO_FAILURE = "A02:2021-Cryptographic Failures"
    BROKEN_AUTH = "A03:2021-Broken Authentication"
    INSECURE_DESIGN = "A04:2021-Insecure Design"
    SECURITY_MISC = "A05:2021-Security Misconfiguration"
    VULNERABLE_COMPONENTS = "A06:2021-Vulnerable Components"
    AUTH_FAILURE = "A07:2021-Auth Failure"
    DATA_INTEGRITY = "A08:2021-Data Integrity Failures"
    SECURITY_LOGGING = "A09:2021-Security Logging Failures"
    SSRF = "A10:2021-SSRF"


class VulnerabilityFinding(BaseModel):
    """
    Pydantic model for LangChain structured output parsing.

    Represents a single vulnerability finding with details such as type,
    severity, location, impact, and recommendations for mitigation.

    Attributes:
        vulnerability_type (VulnerabilityType): OWASP vulnerability type.
        file_path (Optional[str]): File path containing the vulnerability.
        line_number (int): Line number of the vulnerability.
        code_snippet (str): The exact vulnerable code snippet.
        description (str): Description of the vulnerability and its impact.
        severity (str): Severity level (High/Medium/Low).
        recommendation (str): Recommendations for fixing the vulnerability.
        corrected_code (str): Corrected version of the vulnerable code.
        cwe_id (Optional[str]): CWE ID, if applicable.
        risk_score (float): Risk score (0-10) based on severity and impact.
        impact (str): Detailed impact analysis.
    """

    vulnerability_type: VulnerabilityType = Field(
        ..., description="The type of vulnerability from OWASP Top 10 2021"
    )
    file_path: Optional[str] = Field(
        None, description="Path to the file containing vulnerability"
    )
    line_number: int = Field(
        ..., description="Line number where the vulnerability was found"
    )
    code_snippet: str = Field(..., description="The exact vulnerable code snippet")
    description: str = Field(
        ...,
        description="Detailed description of the vulnerability and its potential impact",
    )
    severity: str = Field(
        ..., description="Severity level of the vulnerability (High/Medium/Low)"
    )
    recommendation: str = Field(
        ..., description="Specific recommendations to fix the vulnerability"
    )
    corrected_code: str = Field(
        ..., description="The corrected version of the vulnerable code"
    )
    cwe_id: Optional[str] = Field(
        None, description="CWE ID if applicable (e.g., CWE-89 for SQL Injection)"
    )
    risk_score: float = Field(
        ..., description="Risk score from 0-10 based on severity and impact"
    )
    impact: str = Field(
        ..., description="Detailed impact analysis of the vulnerability"
    )


class VulnerabilityList(BaseModel):
    """
    Pydantic model for a list of vulnerabilities.

    Encapsulates multiple vulnerability findings in a single structure
    for processing and reporting.

    Attributes:
        vulnerabilities (List[VulnerabilityFinding]): List of detected vulnerabilities.
    """

    vulnerabilities: List[VulnerabilityFinding] = Field(
        ..., description="List of vulnerabilities found in the code"
    )


class SecurityScanResult(BaseModel):
    """
    Model for scan results including vulnerabilities and metrics.

    Represents the output of a security scan, including detected vulnerabilities,
    performance metrics, and a summarized scan report.

    Attributes:
        vulnerabilities (List[VulnerabilityFinding]): Detected vulnerabilities.
        metrics (Dict[str, Any]): Metrics such as timing, line counts, etc.
        scan_summary (Dict[str, Any]): Summary of the scan results.
    """

    vulnerabilities: List[VulnerabilityFinding]
    metrics: Dict[str, Any]
    scan_summary: Dict[str, Any]


class ChartDrawing(Flowable):
    """Custom flowable for drawing charts"""

    def __init__(self, width, height):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def draw(self):
        """Draw the chart"""
        self.canv.setStrokeColor(colors.black)
        self.canv.setFillColor(colors.black)
