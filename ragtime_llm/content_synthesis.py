"""
Content synthesis module for deep-dive analysis of topics and creators.

This module provides functionality for:
1. Topic deep-dive: Analyzing and synthesizing content across multiple videos about a specific topic
2. Creator deep-dive: Analyzing a creator's body of work to identify patterns, themes, and insights
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from collections import defaultdict

from ragtime_llm.core.unified_rag_system import YouTubeRAG
from ragtime_llm.utils.logger import logger

class ContentSynthesizer:
    """Handles deep-dive analysis of topics and creators."""

    def __init__(self, rag_system: YouTubeRAG):
        """Initialize the content synthesizer.
        
        Args:
            rag_system: The RAG system instance to use for content analysis
        """
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)

    def topic_deep_dive(
        self,
        topic: str,
        max_videos: int = 10,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict:
        """Perform a deep-dive analysis of a specific topic across multiple videos.
        
        Args:
            topic: The topic to analyze
            max_videos: Maximum number of videos to include in analysis
            max_tokens: Maximum tokens for the response
            temperature: Temperature for response generation
            
        Returns:
            Dict containing:
            - summary: Overall summary of the topic
            - key_points: List of key points across videos
            - timeline: Chronological development of the topic
            - sources: List of video sources used
        """
        try:
            # Query for relevant content
            query = f"Analyze and synthesize content about {topic}"
            response = self.rag_system.generate_response(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract key points and timeline
            key_points_query = f"What are the key points about {topic} across these videos?"
            key_points = self.rag_system.generate_response(
                query=key_points_query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Generate timeline
            timeline_query = f"Create a chronological timeline of how {topic} has evolved across these videos"
            timeline = self.rag_system.generate_response(
                query=timeline_query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "summary": response,
                "key_points": key_points,
                "timeline": timeline,
                "sources": self.rag_system.get_recent_sources(max_videos)
            }
            
        except Exception as e:
            self.logger.error(f"Error in topic deep-dive: {e}")
            raise

    def creator_deep_dive(
        self,
        creator_url: str,
        max_videos: int = 20,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> Dict:
        """Perform a deep-dive analysis of a creator's body of work.
        
        Args:
            creator_url: URL of the creator's channel
            max_videos: Maximum number of videos to analyze
            max_tokens: Maximum tokens for the response
            temperature: Temperature for response generation
            
        Returns:
            Dict containing:
            - creator_summary: Overall analysis of the creator
            - content_themes: Main themes in their content
            - style_analysis: Analysis of their presentation style
            - evolution: How their content has evolved
            - sources: List of videos analyzed
        """
        try:
            # Add creator's videos to the system
            self.rag_system.add_creator_videos(creator_url, max_videos=max_videos)
            
            # Analyze creator's content
            creator_query = f"Analyze this creator's body of work and provide insights"
            creator_summary = self.rag_system.generate_response(
                query=creator_query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Identify content themes
            themes_query = "What are the main themes and topics in this creator's content?"
            content_themes = self.rag_system.generate_response(
                query=themes_query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Analyze presentation style
            style_query = "Analyze this creator's presentation style and techniques"
            style_analysis = self.rag_system.generate_response(
                query=style_query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Analyze content evolution
            evolution_query = "How has this creator's content evolved over time?"
            evolution = self.rag_system.generate_response(
                query=evolution_query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "creator_summary": creator_summary,
                "content_themes": content_themes,
                "style_analysis": style_analysis,
                "evolution": evolution,
                "sources": self.rag_system.get_recent_sources(max_videos)
            }
            
        except Exception as e:
            self.logger.error(f"Error in creator deep-dive: {e}")
            raise

    def generate_report(
        self,
        analysis: Dict,
        output_format: str = "markdown"
    ) -> str:
        """Generate a formatted report from the analysis.
        
        Args:
            analysis: The analysis results from topic_deep_dive or creator_deep_dive
            output_format: Format of the report (markdown or html)
            
        Returns:
            Formatted report as a string
        """
        if output_format == "markdown":
            return self._generate_markdown_report(analysis)
        elif output_format == "html":
            return self._generate_html_report(analysis)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_markdown_report(self, analysis: Dict) -> str:
        """Generate a markdown report from the analysis."""
        report = []
        
        # Add timestamp
        report.append(f"# Analysis Report\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add sections based on analysis type
        if "summary" in analysis:  # Topic deep-dive
            report.extend([
                "## Summary",
                analysis["summary"],
                "\n## Key Points",
                analysis["key_points"],
                "\n## Timeline",
                analysis["timeline"],
                "\n## Sources",
                *[f"- {source}" for source in analysis["sources"]]
            ])
        else:  # Creator deep-dive
            report.extend([
                "## Creator Analysis",
                analysis["creator_summary"],
                "\n## Content Themes",
                analysis["content_themes"],
                "\n## Style Analysis",
                analysis["style_analysis"],
                "\n## Content Evolution",
                analysis["evolution"],
                "\n## Sources",
                *[f"- {source}" for source in analysis["sources"]]
            ])
        
        return "\n".join(report)

    def _generate_html_report(self, analysis: Dict) -> str:
        """Generate an HTML report from the analysis."""
        # Basic HTML template
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<style>",
            "body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "h1, h2 { color: #333; }",
            "ul { list-style-type: none; padding-left: 0; }",
            "li { margin-bottom: 10px; }",
            "</style>",
            "</head>",
            "<body>"
        ]
        
        # Add timestamp
        html.append(f"<h1>Analysis Report</h1>")
        html.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Add sections based on analysis type
        if "summary" in analysis:  # Topic deep-dive
            html.extend([
                "<h2>Summary</h2>",
                f"<p>{analysis['summary']}</p>",
                "<h2>Key Points</h2>",
                f"<p>{analysis['key_points']}</p>",
                "<h2>Timeline</h2>",
                f"<p>{analysis['timeline']}</p>",
                "<h2>Sources</h2>",
                "<ul>",
                *[f"<li>{source}</li>" for source in analysis["sources"]],
                "</ul>"
            ])
        else:  # Creator deep-dive
            html.extend([
                "<h2>Creator Analysis</h2>",
                f"<p>{analysis['creator_summary']}</p>",
                "<h2>Content Themes</h2>",
                f"<p>{analysis['content_themes']}</p>",
                "<h2>Style Analysis</h2>",
                f"<p>{analysis['style_analysis']}</p>",
                "<h2>Content Evolution</h2>",
                f"<p>{analysis['evolution']}</p>",
                "<h2>Sources</h2>",
                "<ul>",
                *[f"<li>{source}</li>" for source in analysis["sources"]],
                "</ul>"
            ])
        
        html.extend(["</body>", "</html>"])
        return "\n".join(html) 