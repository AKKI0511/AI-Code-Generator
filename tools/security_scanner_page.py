import shutil
from typing import List
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from models.vulnerability_scanner.helper import SecurityScanResult, VulnerabilityFinding
from models.vulnerability_scanner.vulnerability_scanner import SecurityScanner
from streamlit.runtime.uploaded_file_manager import UploadedFile as UploadFile

class SecurityScannerPage:
    """Security Scanner Page for the Streamlit application.

    This class handles the rendering and functionality of the security scanner page,
    including scan options, file uploads, git repository scanning, and displaying scan results.
    """

    def __init__(self):
        """Initialize the SecurityScannerPage with a SecurityScanner instance and configure the page."""
        self.scanner = SecurityScanner(st.session_state.assistant)
        self._configure_page()

    def _configure_page(self):
        """Configure the Streamlit page settings and apply custom CSS styles."""
        st.title("🔒 Advanced Code Security Scanner")
        st.markdown(
            """
            <style>
            .vulnerability-card {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 1rem;
                margin: 0.5rem 0;
                background-color: rgba(255, 255, 255, 0.05);
            }
            .severity-high { border-left: 4px solid #ff4b4b; }
            .severity-medium { border-left: 4px solid #ffa500; }
            .severity-low { border-left: 4px solid #00cc00; }
            .metric-card {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
            }
            .recommendation-card {
                background-color: rgba(0, 100, 255, 0.1);
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render_scan_options(self):
        """Render the scan options section in the sidebar."""
        with st.sidebar:
            st.markdown("---")
            st.header("Scan Options")

            # Scan Type Selection
            scan_type = st.radio(
                "Select Scan Type",
                ["Quick Scan", "Deep Scan", "Custom Scan"],
                help="Quick Scan: Basic security checks\nDeep Scan: Comprehensive analysis\nCustom Scan: Configure specific checks",
            )

            if scan_type == "Custom Scan":
                selected_checks = st.multiselect(
                    "Select Security Checks",
                    [
                        "SQL Injection",
                        "XSS",
                        "CSRF",
                        "Authentication",
                        "Authorization",
                        "Input Validation",
                        "File Operations",
                        "Network Security",
                    ],
                    default=["SQL Injection", "XSS"],
                    key="custom_checks",
                )
                st.session_state.selected_security_checks = selected_checks

            # Scan Mode Selection
            option = st.selectbox(
                "Input Method", ("Upload Files", "Scan Code Snippet", "Git Repository")
            )

            if option == "Upload Files":
                self._render_file_upload()
            elif option == "Scan Code Snippet":
                self._render_code_input()
            else:
                self._render_git_input()

    def _render_git_input(self):
        """Render the git repository input section for scanning."""
        st.header("Git Repository Scanning")
        repo_url = st.text_input(
            "Git Repository URL", 
            placeholder="https://github.com/username/repo"
        ).strip()
        
        # Validate URL format
        if repo_url and not (repo_url.startswith('http://') or repo_url.startswith('https://')):
            st.error("Please enter a valid HTTP/HTTPS Git repository URL.")
            return

        branch = st.text_input("Branch (optional)", value="main").strip()
        auth_method = st.selectbox(
            "Authentication Method", 
            ("None", "Username & Password", "Personal Access Token")
        )

        credentials = {}
        if auth_method == "Username & Password":
            credentials['username'] = st.text_input("Username").strip()
            credentials['password'] = st.text_input("Password", type="password").strip()
            if not (credentials['username'] and credentials['password']):
                st.warning("Both username and password are required.")
                return
        elif auth_method == "Personal Access Token":
            credentials['token'] = st.text_input("Access Token", type="password").strip()
            if not credentials['token']:
                st.warning("Access token is required.")
                return

        if st.button("Scan Repository"):
            if not repo_url:
                st.error("Please enter a valid Git repository URL.")
                return

            temp_dir = Path("temp_git_repo")
            try:
                # Clean up any existing directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Prepare git clone command
                if auth_method == "Username & Password":
                    # Extract domain from URL
                    domain = repo_url.split('https://')[-1]
                    git_url = f"https://{credentials['username']}:{credentials['password']}@{domain}"
                elif auth_method == "Personal Access Token":
                    domain = repo_url.split('https://')[-1]
                    git_url = f"https://{credentials['token']}@{domain}"
                else:
                    git_url = repo_url

                clone_command = ["git", "clone"]
                if branch:
                    clone_command.extend(["--branch", branch, "--single-branch"])
                clone_command.extend([git_url, str(temp_dir)])

                with st.spinner("Cloning repository..."):
                    import subprocess
                    result = subprocess.run(
                        clone_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300  # 5-minute timeout
                    )

                    if result.returncode != 0:
                        st.error(f"Failed to clone repository: {result.stderr}")
                        return

                    st.success("Repository cloned successfully.")

                with st.spinner("Scanning repository for vulnerabilities..."):
                    scan_result = self.scanner.scan_files(str(temp_dir))
                    self._save_scan_result(
                        scan_result, 
                        f"Git Repo Scan: {repo_url.split('/')[-1]}@{branch or 'main'}"
                    )
                    st.success("Repository scanned successfully.")

            except subprocess.TimeoutExpired:
                st.error("Repository cloning timed out. Please try again with a smaller repository.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Cleanup using shutil for reliable directory removal
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    st.warning(f"Could not fully clean up temporary files: {str(e)}")

    def _save_scan_result(self, scan_result: SecurityScanResult, scan_name: str):
        """Save the scan result to the session state with metadata.

        Args:
            scan_result (SecurityScanResult): The result of the security scan.
            scan_name (str): The name of the scan.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scan_id = len(st.session_state.scan_history)

            if not hasattr(scan_result, 'vulnerabilities'):
                raise ValueError("Invalid scan result format")

            scan_data = {
                "id": scan_id,
                "name": scan_name,
                "timestamp": timestamp,
                "result": scan_result,
                "stats": self._calculate_scan_stats(scan_result),
            }

            st.session_state.scan_history.append(scan_data)
            st.session_state.current_scan_id = scan_id
            st.session_state["scan_result"] = scan_result

            st.balloons()
            st.success(f"Scan '{scan_name}' completed and saved to history.")
        except Exception as e:
            st.error(f"Failed to save scan results: {str(e)}")

    def _calculate_scan_stats(self, scan_result: SecurityScanResult):
        """Calculate statistics for a given scan result.

        Args:
            scan_result (SecurityScanResult): The result of the security scan.

        Returns:
            dict: A dictionary containing the total number of vulnerabilities, counts of high, medium, and low severity vulnerabilities, and the average risk score.
        """
        vulnerabilities = scan_result.vulnerabilities
        return {
            "total": len(vulnerabilities),
            "high": sum(1 for v in vulnerabilities if v.severity == "High"),
            "medium": sum(1 for v in vulnerabilities if v.severity == "Medium"),
            "low": sum(1 for v in vulnerabilities if v.severity == "Low"),
            "risk_score": (
                sum(v.risk_score for v in vulnerabilities) / len(vulnerabilities)
                if vulnerabilities
                else 0
            ),
        }

    def _render_file_upload(self):
        """Render the file upload section for scanning code files."""
        uploaded_files = st.file_uploader(
            "Upload Code Files",
            accept_multiple_files=True,
            type=["py", "js", "ts", "java", "cpp", "c", "php", "rb", "go"],
        )

        if uploaded_files and st.button("Start Scan"):
            with st.spinner("Scanning files for vulnerabilities..."):
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                file_paths = [
                    self._save_file(temp_dir, file) for file in uploaded_files
                ]

                try:
                    scan_result = self.scanner.scan_files(str(temp_dir))
                    self._save_scan_result(
                        scan_result, f"File Scan: {len(uploaded_files)} files"
                    )
                finally:
                    # Cleanup
                    for file_path in file_paths:
                        Path(file_path).unlink()
                    temp_dir.rmdir()

    def _save_file(self, temp_dir: Path, file: UploadFile):
        """Save an uploaded file to a temporary directory.

        Args:
            temp_dir (Path): The temporary directory to save the file.
            file (UploadFile): The uploaded file.

        Returns:
            Path: The path to the saved file.
        """
        file_path = temp_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())
        return file_path

    def _render_code_input(self):
        """Render the code input section for scanning code snippets."""
        st.header("Enter Code to Scan")
        code = st.text_area("Paste your code here", height=300, key="manual_code_input")
        if st.button("Scan Code"):
            if not code.strip():
                st.warning("Please enter some code to scan.")
            else:
                try:
                    with st.spinner("Scanning the provided code for vulnerabilities..."):
                        scan_result = self.scanner.scan_code(code)
                        self._save_scan_result(scan_result, "Manual Code Input Scan")
                except Exception as e:
                    st.error(f"An error occurred during scanning: {str(e)}")

    def display_scan_results(self, scan_result: SecurityScanResult):
        """Display the results of a security scan.

        Args:
            scan_result (SecurityScanResult): The result of the security scan.
        """
        if not scan_result.vulnerabilities:
            st.success("🎉 No vulnerabilities found in the scan!")
            return

        # Filters
        st.sidebar.header("Filters")
        severities = st.sidebar.multiselect(
            "Filter by Severity",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
            key="filter_severity",
        )

        # Apply filters
        filtered_vulns = [
            v
            for v in scan_result.vulnerabilities
            if hasattr(v, "severity") and v.severity in severities
        ]

        # Dashboard Layout
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            self._render_trend_analysis()

        with col2:
            self._render_severity_distribution(filtered_vulns)

        with col3:
            self._render_risk_metrics(filtered_vulns)

        # Detailed Findings
        st.header("📊 Detailed Findings")
        tabs = st.tabs(["Vulnerabilities", "Timeline", "Recommendations", "Export"])

        with tabs[0]:
            self._render_vulnerability_details(filtered_vulns)

        with tabs[1]:
            self._render_scan_timeline()

        with tabs[2]:
            self._render_recommendations(filtered_vulns)

        with tabs[3]:
            self._render_export_options(scan_result)

    def _render_trend_analysis(self):
        """Render the vulnerability trend analysis chart."""
        if not st.session_state.scan_history:
            return

        trend_data = []
        for scan in st.session_state.scan_history:
            trend_data.append(
                {
                    "timestamp": scan["timestamp"],
                    "high": scan["stats"]["high"],
                    "medium": scan["stats"]["medium"],
                    "low": scan["stats"]["low"],
                }
            )

        df = pd.DataFrame(trend_data)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["high"], name="High", line=dict(color="red")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["medium"],
                name="Medium",
                line=dict(color="orange"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["low"], name="Low", line=dict(color="green")
            )
        )

        fig.update_layout(
            title="Vulnerability Trends Over Time",
            xaxis_title="Scan Date",
            yaxis_title="Number of Vulnerabilities",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_severity_distribution(self, vulnerabilities: List[VulnerabilityFinding]):
        """Render the severity distribution chart.

        Args:
            vulnerabilities (List[VulnerabilityFinding]): A list of vulnerability findings.
        """
        severity_counts = {
            "High": sum(1 for v in vulnerabilities if v.severity == "High"),
            "Medium": sum(1 for v in vulnerabilities if v.severity == "Medium"),
            "Low": sum(1 for v in vulnerabilities if v.severity == "Low"),
        }

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(severity_counts.keys()),
                    values=list(severity_counts.values()),
                    hole=0.3,
                    marker=dict(colors=["red", "orange", "green"]),
                )
            ]
        )

        fig.update_layout(title="Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)

    def _render_risk_metrics(self, vulnerabilities: List[VulnerabilityFinding]):
        """Render key risk metrics.

        Args:
            vulnerabilities (List[VulnerabilityFinding]): A list of vulnerability findings.
        """
        total = len(vulnerabilities)
        risk_score = (
            sum(v.risk_score for v in vulnerabilities) / total if total > 0 else 0
        )

        st.metric("Total Vulnerabilities", total)
        st.metric("Average Risk Score", f"{risk_score:.2f}")
        st.metric(
            "Critical Issues", sum(1 for v in vulnerabilities if v.severity == "High")
        )

    def _render_vulnerability_details(self, vulnerabilities: List[VulnerabilityFinding]):
        """Render detailed vulnerability information.

        Args:
            vulnerabilities (List[VulnerabilityFinding]): A list of vulnerability findings.
        """
        for vuln in vulnerabilities:
            severity_class = f"severity-{vuln.severity.lower()}"
            color = (
                "red"
                if vuln.severity == "High"
                else "orange" if vuln.severity == "Medium" else "green"
            )
            st.markdown(
                f"""
                <div class="vulnerability-card {severity_class}" style="border-left: 4px solid {color};">
                    <h3>{vuln.vulnerability_type}</h3>
                    <p><strong>Severity:</strong> {vuln.severity}</p>
                    <p><strong>Risk Score:</strong> {vuln.risk_score:.2f}</p>
                    <p><strong>Location:</strong> {vuln.file_path}:{vuln.line_number}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Show Details"):
                st.markdown("#### Vulnerable Code")
                st.code(vuln.code_snippet, language="python")
                st.markdown("#### Recommended Fix")
                st.code(vuln.corrected_code, language="python")
                st.markdown("#### Impact Analysis")
                st.write(vuln.impact)

    def _render_scan_timeline(self):
        """Render the scan history timeline."""
        if not st.session_state.scan_history:
            st.info("No scan history available")
            return

        for scan in reversed(st.session_state.scan_history):
            st.markdown(
                f"""
                <div class="vulnerability-card">
                    <h4>{scan['name']}</h4>
                    <p>Scanned on: {scan['timestamp']}</p>
                    <p>Found: {scan['stats']['total']} vulnerabilities</p>
                    <p>Average Risk Score: {scan['stats']['risk_score']:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_recommendations(self, vulnerabilities: List[VulnerabilityFinding]):
        """Render security recommendations based on the scan results.

        Args:
            vulnerabilities (List[VulnerabilityFinding]): A list of vulnerability findings.
        """
        recommendations = {}
        for vuln in vulnerabilities:
            if vuln.recommendation not in recommendations:
                recommendations[vuln.recommendation] = {
                    "count": 0,
                    "severity": vuln.severity,
                    "type": vuln.vulnerability_type,
                }
            recommendations[vuln.recommendation]["count"] += 1

        for rec, details in recommendations.items():
            st.markdown(
                f"""
                <div class="recommendation-card">
                    <h4>{details['type']} ({details['count']} occurrences)</h4>
                    <p>{rec}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    def _render_export_options(self, scan_result: SecurityScanResult):
        """Render options to export scan results.

        Args:
            scan_result (SecurityScanResult): The result of the security scan.
        """
        st.subheader("Export Scan Results")

        export_format = st.selectbox("Select Export Format", ["JSON", "PDF"])

        if st.button("Download"):
            try:
                # Generate the report file
                output_file_path = self.scanner.generate_report(scan_result, export_format.lower())
                
                # Read file content
                with open(output_file_path, "rb") as file:
                    file_data = file.read()
                
                # Configure download button
                file_name = f"scan_results.{export_format.lower()}"
                mime_type = "application/json" if export_format == "JSON" else "application/pdf"

                st.download_button(
                    label=f"Download {export_format}",
                    data=file_data,
                    file_name=file_name,
                    mime=mime_type,
                )
            except Exception as e:
                st.error(f"An error occurred while exporting: {str(e)}")

    def render(self):
        """Main render method for the security scanner page.

        This method renders the scan options and displays the scan results if available.
        """
        self.render_scan_options()

        if "scan_result" in st.session_state and isinstance(
            st.session_state["scan_result"], SecurityScanResult
        ):
            self.display_scan_results(st.session_state["scan_result"])
        else:
            st.info("Run a security scan to see the results here.")
