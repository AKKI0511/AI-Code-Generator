import re
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, UTC
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import CSS, HTML
import logging
import aiofiles
import asyncio
import json
import os

from .utils import format_size, create_error_report, measure_execution_time
from .config import ExecutionConfig, ReportConfig, get_config, Config

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Pydantic models for structured output
class OptimizedCode(BaseModel):
    """Code optimization details"""
    original: str = Field(description="Original code")
    optimized: str = Field(description="Optimized code")
    impact_summary: str = Field(description="Performance impact")
    reasoning: Optional[str] = Field(None, description="Optimization rationale")

class PerformanceIssue(BaseModel):
    """Performance issue with solution"""
    location: str = Field(description="Issue location")
    issue_type: str = Field(description="Issue type")
    priority_level: str = Field(description="high|medium|low")
    solution: OptimizedCode = Field(description="Solution details")

class ResourceMetrics(BaseModel):
    """Essential resource metrics"""
    peak_memory_mb: float = Field(description="Peak memory usage (MB)")
    avg_cpu_percent: float = Field(description="Average CPU usage (%)")
    exec_time_ms: float = Field(description="Execution time (ms)")

class ProfileReport(BaseModel):
    """Performance profiling report"""
    issues: List[PerformanceIssue] = Field(
        default_factory=list,
        description="Performance issues"
    )
    metrics: ResourceMetrics = Field(
        description="Resource metrics"
    )

class AnalysisResponse(BaseModel):
    """Combined narrative and structured analysis response"""
    narrative: str = Field(description="Narrative analysis and recommendations")
    profile_report: ProfileReport = Field(description="Structured performance report")


class MetricsPreprocessor:
    """Preprocesses profiling metrics to optimize LLM token usage"""
    
    @staticmethod
    def filter_relevant_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only the most relevant metrics for analysis"""
        return {
            "code": metrics.get("code", ""),
            "execution_time": metrics.get("execution", {}).get("time"),
            "memory_usage": {
                "used_mb": metrics.get("memory", {}).get("used", 0) / (1024 * 1024),
                "percent": metrics.get("memory", {}).get("percent", 0)
            },
            "cpu_usage": metrics.get("cpu", {}),
            "hotspots": MetricsPreprocessor._extract_hotspots(
                metrics.get("profiled_functions", {})
            )
        }
    
    @staticmethod
    def _extract_hotspots(profiled_functions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract performance hotspots from profiled functions"""
        hotspots = []
        for func_name, data in profiled_functions.items():
            if "line_profile" in data:
                hotspots.extend([
                    {
                        "function": func_name,
                        "line": line_num,
                        "hits": info["hits"],
                        "time_us": info["time_microseconds"]
                    }
                    for line_num, info in data["line_profile"].items()
                    if info["time_microseconds"] > 1000  # Focus on significant hotspots
                ])
        return sorted(hotspots, key=lambda x: x["time_us"], reverse=True)[:5]


class ProfilerReportGenerator:
    """
    Generates comprehensive profiling reports using LLM-enhanced analysis.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all components required for report generation"""
        self._setup_llm()
        self._setup_templating()
        self._setup_output_directory()

    def _setup_llm(self) -> None:
        """Configure LLM components with optimization-focused prompt"""
        self.api_key = os.getenv("GROQ_API_KEY") or self.config.env_vars.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found")

        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.config.report.llm_config["model"],
            temperature=self.config.report.llm_config["temperature"],
            max_tokens=self.config.report.llm_config["max_tokens"],
        )

        self.output_parser = PydanticOutputParser(pydantic_object=ProfileReport)
        
        # Optimization-focused prompt template
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python performance optimizer. Your response should have two parts:

1. NARRATIVE: A brief (max 200 words) analysis highlighting key findings and recommendations.
2. JSON_REPORT: A structured JSON report matching the ProfileReport schema.

Use exactly these markers to separate the sections:
---NARRATIVE START---
(your narrative here)
---NARRATIVE END---
---JSON START---
(your JSON report here)
---JSON END---"""),
            ("user", """Analyze these performance metrics and provide optimizations:
{metrics}

Focus on actionable improvements and quantifiable gains.
{format_instructions}""")
        ])

    def _setup_templating(self) -> None:
        """Initialize Jinja2 environment with custom filters"""
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.config.report.template_dir),
            autoescape=select_autoescape(["html", "xml"])
        )
        
        # Add custom filters for formatting
        self.jinja_env.filters["format_datetime"] = lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        self.jinja_env.filters["format_size"] = format_size

        # Set up CSS path relative to the report output directory
        self.css_path = Path(self.config.report.template_dir) / "performance_report.css"
        self.logger.info(f"CSS file path: {self.css_path}")
        if not self.css_path.exists():
            raise FileNotFoundError(f"CSS file not found at {self.css_path}")

    def _setup_output_directory(self) -> None:
        """Create output directory for reports"""
        self.output_dir = Path(self.config.report.report_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _parse_llm_response(self, response: str) -> AnalysisResponse:
        """Parse LLM response into narrative and structured components"""
        # Extract narrative
        narrative_match = re.search(
            r"---NARRATIVE START---\s*(.*?)\s*---NARRATIVE END---",
            response,
            re.DOTALL
        )
        
        # Extract JSON
        json_match = re.search(
            r"---JSON START---\s*(.*?)\s*---JSON END---",
            response,
            re.DOTALL
        )
        
        if not narrative_match or not json_match:
            raise ValueError("Failed to parse LLM response: Missing required sections")
            
        narrative = narrative_match.group(1).strip()
        json_str = json_match.group(1).strip()
        
        # Parse JSON into ProfileReport
        try:
            profile_report = ProfileReport(**json.loads(json_str))
        except Exception as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"LLM response:\n{'='*50}\n{response}")
            raise ValueError(f"Invalid JSON structure: {e}")
            
        return AnalysisResponse(
            narrative=narrative,
            profile_report=profile_report
        )
    
    async def _generate_llm_prompt(self, metrics: Dict[str, Any]) -> str:
        """Generate optimized LLM prompt from preprocessed metrics"""
        preprocessed_metrics = MetricsPreprocessor.filter_relevant_metrics(metrics)
        return self.analysis_prompt.format(
            metrics=json.dumps(preprocessed_metrics, indent=2),
            format_instructions=self.output_parser.get_format_instructions()
        )
    
    async def _invoke_llm(self, prompt: str) -> AnalysisResponse:
        """Send prompt to LLM and parse response with error handling"""
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=self.config.execution.max_execution_time
            )
            
            return await self._parse_llm_response(response.content)
            
        except asyncio.TimeoutError:
            self.logger.error("LLM request timed out")
            raise
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {e}")
            raise

    @measure_execution_time
    async def generate_report(
        self, metrics: Dict[str, Any], formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive profiling reports in multiple formats.

        Args:
            metrics: Raw profiling metrics
            formats: List of desired formats ("json", "html", "md", "pdf")

        Returns:
            Dictionary mapping format to report content
        """
        try:
            # Default to all formats if none specified
            formats = formats or ["json", "html", "md", "pdf"]

            # Get LLM analysis
            prompt = await self._generate_llm_prompt(metrics)
            analysis = await self._invoke_llm(prompt)
            
            # Generate reports in requested formats
            reports = {}
            
            for format_type in formats:
                if format_type == "json":
                    reports["json"] = self._generate_json_report(metrics, analysis)
                elif format_type == "html":
                    reports["html"] = self._generate_html_report(metrics, analysis)
                elif format_type == "md":
                    reports["md"] = self._generate_markdown_report(metrics, analysis)
                elif format_type == "pdf":
                    if "html" not in reports:
                        reports["html"] = self._generate_html_report(metrics, analysis)
                    reports["pdf"] = self._generate_pdf_report(reports["html"])

            if self.config.report.save_reports:
                await self._save_reports(reports)

            return {
                'narrative': analysis.narrative,
                'profile_report': analysis.profile_report.model_dump(),
                'reports': reports,
            }

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise

    def _generate_json_report(
        self, 
        metrics: Dict[str, Any], 
        analysis: AnalysisResponse
    ) -> str:
        """Generate JSON format report"""
        report_data = {
            "metrics_summary": MetricsPreprocessor.filter_relevant_metrics(metrics),
            "narrative_analysis": analysis.narrative,
            "performance_analysis": analysis.profile_report.model_dump(),
            "metadata": {
                "generated_at": datetime.now(UTC).isoformat(),
                "version": self.config.report.report_format_version
            }
        }
        return json.dumps(report_data, indent=2)

    def _generate_html_report(
        self, 
        metrics: Dict[str, Any], 
        analysis: AnalysisResponse
    ) -> str:
        """Generate HTML format report using Jinja2"""
        template = self.jinja_env.get_template("performance_report.html")
        return template.render(
            metrics=MetricsPreprocessor.filter_relevant_metrics(metrics),
            narrative=analysis.narrative,
            analysis=analysis.profile_report,
            generated_at=datetime.now(UTC),
            css_path=str(self.css_path)
        )

    def _generate_markdown_report(
        self, 
        metrics: Dict[str, Any], 
        analysis: AnalysisResponse
    ) -> str:
        """Generate Markdown format report using Jinja2"""
        template = self.jinja_env.get_template("performance_report.md")
        return template.render(
            metrics=MetricsPreprocessor.filter_relevant_metrics(metrics),
            narrative=analysis.narrative,
            analysis=analysis.profile_report,
            generated_at=datetime.now(UTC)
        )

    def _generate_pdf_report(self, html_content: str) -> bytes:
        """Convert HTML report to PDF using WeasyPrint with proper CSS handling"""
        try:
            # Create a temporary HTML file with proper CSS path
            with tempfile.NamedTemporaryFile(suffix='.html', mode='w+', encoding='utf-8', delete=False) as temp_html:
                temp_html.write(html_content)
                temp_html_path = temp_html.name

            # Initialize WeasyPrint HTML with base URL for proper CSS resolution
            html = HTML(filename=temp_html_path, base_url=str(self.config.report.template_dir))
            
            # Generate PDF with CSS
            pdf_bytes = html.write_pdf(presentational_hints=True)

            return pdf_bytes

        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            raise

        finally:
            # Clean up temporary file
            if 'temp_html_path' in locals():
                try:
                    os.unlink(temp_html_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary HTML file: {e}")
    
    async def _save_reports(self, reports: Dict[str, Any]) -> None:
        """Save reports to disk with error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for format_type, content in reports.items():
                filename = f"profiled_LLM_report_{timestamp}.{format_type}"
                filepath = self.output_dir / filename
                
                mode = "wb" if isinstance(content, bytes) else "w"
                # Specify encoding for text files
                async with aiofiles.open(filepath, mode=mode, encoding='utf-8' if mode == 'w' else None) as f:
                    await f.write(content)
            
            self.logger.info(f"Profiled LLM reports saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save reports: {e}")
            raise

# Example usage
if __name__ == "__main__":

    async def run_example() -> None:
        """
        Demonstrates the usage of ProfilerReportGenerator with sample performance data.
        Generates reports in multiple formats and handles various scenarios.
        """
        # Sample performance metrics data
        sample_metrics: Dict[str, Any] = {
            "code": """
def process_data(items: list) -> list:
    result = []
    for item in items:
        result.append(item ** 2)
    return result

def analyze_numbers(count: int) -> tuple:
    numbers = list(range(count))
    processed = process_data(numbers)
    total = sum(processed)
    average = total / len(processed)
    return total, average

# Main execution
result = analyze_numbers(1_000_000)
print(f"Analysis result: {result}")
""",
            "execution": {
                "time": 1.547752857208252,
                "success": True,
                "namespace": {
                    'process_data': '<function process_data at 0x000001D85BE46020>',
                    'analyze_numbers': '<function analyze_numbers at 0x000001D85BE46080>',
                    'result': (333332833333500000, 333332.8333335)
                },
                "execution_context": {
                    "sandbox_enabled": True,
                    "timeout_enabled": True,
                },
            },
            "memory": {
                "used": 13144993792,
                "percent": 78.1,
                "swap_percent": 5.0
            },
            "cpu": {
                "percent": 6.1,
                "process_percent": 0.0,
                "cpu_cores_utilized": 2
            },
            "profiled_functions": {
                "process_data": {
                    "line_profile": {
                        "2": {
                            "hits": 1,
                            "time_microseconds": 16,
                            "function": "process_data",
                        },
                        "3": {
                            "hits": 1000000,
                            "time_microseconds": 3582287,
                            "function": "process_data",
                        },
                        "4": {
                            "hits": 1000000,
                            "time_microseconds": 2536776,
                            "function": "process_data",
                        },
                    },
                    "duration": 0.847752857208252,
                    "memory_delta": 8128832,
                    "cpu_percent": 0.0,
                },
                "analyze_numbers": {
                    "line_profile": {
                        "7": {
                            "hits": 1,
                            "time_microseconds": 285565,
                            "function": "analyze_numbers",
                        },
                        "8": {
                            "hits": 1,
                            "time_microseconds": 850776,
                            "function": "analyze_numbers",
                        },
                        "9": {
                            "hits": 1,
                            "time_microseconds": 395287,
                            "function": "analyze_numbers",
                        },
                        "10": {
                            "hits": 1,
                            "time_microseconds": 16398,
                            "function": "analyze_numbers",
                        },
                    },
                    "duration": 0.7,
                    "memory_delta": 3000000,
                    "cpu_percent": 0.0,
                }
            },
            "system_info": {
                "python_version": "3.11.9",
                "platform": "Windows-10-10.0.22631-SP0",
                "cpu_cores": 20,
            },
        }

        try:
            # Initialize configuration
            config = Config(
                report=ReportConfig(
                    template_dir="templates",
                    report_output_dir="reports",
                    llm_config={
                        "model": "llama-3.1-8b-instant",
                        "temperature": 0.2,
                        "max_tokens": 4096,
                    },
                    report_format_version="2.0.0",
                    save_reports=True
                ),
                execution=ExecutionConfig(
                    max_execution_time=30
                ),
                env_vars={"GROQ_API_KEY": os.getenv("GROQ_API_KEY")},
            )

            # Initialize the report generator
            generator = ProfilerReportGenerator(config=config)
            
            # Generate reports in all available formats
            reports = await generator.generate_report(
                metrics=sample_metrics,
                formats=["json", "html", "md", "pdf"]
            )

            # Example of accessing and using the generated reports
            if "json" in reports:
                json_data = json.loads(reports["json"])
                print("\nKey Performance Metrics:")
                print(f"Peak Memory Usage: {json_data['metrics_summary']['memory_usage']['used_mb']:.2f} MB")
                print(f"Average CPU Usage: {json_data['metrics_summary']['cpu_usage']['percent']}%")
                print(f"Total Execution Time: {json_data['metrics_summary']['execution_time']:.3f}s")
            if "md" in reports:
                print("="*50, "\n", reports['md'])

        except ValueError as ve:
            print(f"Configuration/Validation Error: {ve}")
        except Exception as e:
            print(f"Error generating report: {e}")
            print(f"Detailed error report: {create_error_report(e)}")

    # Verify environment and run the example
    if os.getenv("GROQ_API_KEY"):
        asyncio.run(run_example())
    else:
        print("Error: GROQ_API_KEY environment variable not set")
        print("Please set the environment variable before running the script:")
        print("export GROQ_API_KEY='your-api-key-here'  # Unix/Linux/macOS")
        print("set GROQ_API_KEY=your-api-key-here  # Windows CMD")
        print("$env:GROQ_API_KEY='your-api-key-here'  # Windows PowerShell")