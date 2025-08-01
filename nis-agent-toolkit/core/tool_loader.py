#!/usr/bin/env python3
"""
NIS Agent Toolkit - Tool Loader System
Dynamic tool loading and management for NIS agents
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional, Type
from dataclasses import dataclass
from rich.console import Console
import logging

console = Console()

@dataclass
class ToolInfo:
    """Information about a loaded tool"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    category: str = "general"
    version: str = "1.0.0"
    author: str = "unknown"

class ToolLoader:
    """
    Dynamic tool loading system for NIS agents
    Honest tool management - no hype, just practical tool integration
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        self.logger = logging.getLogger("nis.tools")
        
        # Load built-in tools
        self._load_builtin_tools()
    
    def _load_builtin_tools(self):
        """Load built-in tools that come with the toolkit"""
        
        # Mathematical tools
        self.register_tool(
            "calculator",
            self._calculator_tool,
            "Basic mathematical calculator",
            {"expression": "str"},
            "math"
        )
        
        self.register_tool(
            "unit_converter",
            self._unit_converter_tool,
            "Convert between units",
            {"value": "float", "from_unit": "str", "to_unit": "str"},
            "math"
        )
        
        # Text processing tools
        self.register_tool(
            "text_analyzer",
            self._text_analyzer_tool,
            "Analyze text properties",
            {"text": "str"},
            "text"
        )
        
        self.register_tool(
            "keyword_extractor",
            self._keyword_extractor_tool,
            "Extract keywords from text",
            {"text": "str", "max_keywords": "int"},
            "text"
        )
        
        # Logic tools
        self.register_tool(
            "logic_checker",
            self._logic_checker_tool,
            "Check logical statements",
            {"statement": "str"},
            "logic"
        )
        
        # Utility tools
        self.register_tool(
            "format_checker",
            self._format_checker_tool,
            "Check data format validity",
            {"data": "any", "format_type": "str"},
            "utility"
        )
        
        self.logger.info(f"Loaded {len(self.tools)} built-in tools")
    
    def register_tool(self, name: str, function: Callable, description: str, 
                     parameters: Dict[str, Any], category: str = "general",
                     version: str = "1.0.0", author: str = "unknown"):
        """Register a new tool"""
        
        tool_info = ToolInfo(
            name=name,
            description=description,
            function=function,
            parameters=parameters,
            category=category,
            version=version,
            author=author
        )
        
        self.tools[name] = tool_info
        
        # Update category index
        if category not in self.tool_categories:
            self.tool_categories[category] = []
        self.tool_categories[category].append(name)
        
        self.logger.info(f"Registered tool: {name} ({category})")
    
    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[ToolInfo]:
        """Get all tools in a category"""
        tool_names = self.tool_categories.get(category, [])
        return [self.tools[name] for name in tool_names]
    
    def list_tools(self) -> Dict[str, List[str]]:
        """List all available tools by category"""
        return self.tool_categories.copy()
    
    def call_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool with parameters"""
        
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found", "success": False}
        
        tool = self.tools[name]
        
        try:
            result = tool.function(**kwargs)
            return {"result": result, "success": True, "tool": name}
        except Exception as e:
            return {"error": str(e), "success": False, "tool": name}
    
    def load_tools_from_directory(self, tools_dir: Path) -> int:
        """Load tools from a directory"""
        
        if not tools_dir.exists():
            self.logger.warning(f"Tools directory not found: {tools_dir}")
            return 0
        
        loaded_count = 0
        
        for py_file in tools_dir.glob("*.py"):
            try:
                loaded_count += self._load_tool_from_file(py_file)
            except Exception as e:
                self.logger.error(f"Error loading tool from {py_file}: {e}")
        
        self.logger.info(f"Loaded {loaded_count} tools from directory")
        return loaded_count
    
    def _load_tool_from_file(self, file_path: Path) -> int:
        """Load tools from a Python file"""
        
        spec = importlib.util.spec_from_file_location("tool_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        loaded_count = 0
        
        # Look for tool functions (functions with 'tool_' prefix)
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and name.startswith("tool_"):
                tool_name = name[5:]  # Remove 'tool_' prefix
                
                # Get tool metadata from docstring or attributes
                description = obj.__doc__ or f"Tool: {tool_name}"
                category = getattr(obj, '__category__', 'custom')
                version = getattr(obj, '__version__', '1.0.0')
                author = getattr(obj, '__author__', 'unknown')
                
                # Extract parameters from function signature
                sig = inspect.signature(obj)
                parameters = {}
                for param_name, param in sig.parameters.items():
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else "any"
                    parameters[param_name] = str(param_type)
                
                self.register_tool(tool_name, obj, description, parameters, category, version, author)
                loaded_count += 1
        
        return loaded_count
    
    # Built-in tool implementations
    def _calculator_tool(self, expression: str) -> Dict[str, Any]:
        """Safe calculator tool"""
        
        try:
            # Basic safety check
            safe_chars = set("0123456789+-*/.() ")
            if not all(c in safe_chars for c in expression):
                return {"error": "Invalid characters in expression", "result": None}
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result, "expression": expression}
            
        except Exception as e:
            return {"error": str(e), "result": None}
    
    def _unit_converter_tool(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert between common units"""
        
        # Simple conversion tables
        length_conversions = {
            "m": 1.0,
            "km": 1000.0,
            "cm": 0.01,
            "mm": 0.001,
            "ft": 0.3048,
            "in": 0.0254
        }
        
        weight_conversions = {
            "kg": 1.0,
            "g": 0.001,
            "lb": 0.453592,
            "oz": 0.0283495
        }
        
        # Determine conversion type
        if from_unit in length_conversions and to_unit in length_conversions:
            conversions = length_conversions
        elif from_unit in weight_conversions and to_unit in weight_conversions:
            conversions = weight_conversions
        else:
            return {"error": "Unsupported unit conversion", "result": None}
        
        try:
            # Convert to base unit, then to target unit
            base_value = value * conversions[from_unit]
            result = base_value / conversions[to_unit]
            
            return {
                "result": result,
                "original_value": value,
                "from_unit": from_unit,
                "to_unit": to_unit
            }
            
        except Exception as e:
            return {"error": str(e), "result": None}
    
    def _text_analyzer_tool(self, text: str) -> Dict[str, Any]:
        """Analyze text properties"""
        
        try:
            words = text.split()
            sentences = text.split('.')
            
            analysis = {
                "character_count": len(text),
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "longest_word": max(words, key=len) if words else "",
                "shortest_word": min(words, key=len) if words else ""
            }
            
            return {"result": analysis, "text": text[:100] + "..." if len(text) > 100 else text}
            
        except Exception as e:
            return {"error": str(e), "result": None}
    
    def _keyword_extractor_tool(self, text: str, max_keywords: int = 10) -> Dict[str, Any]:
        """Extract keywords from text"""
        
        try:
            # Simple keyword extraction
            words = text.lower().split()
            
            # Filter out common words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those"}
            
            keywords = [word for word in words if len(word) > 2 and word not in stop_words]
            
            # Count frequency
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
            
            return {
                "result": [word for word, freq in top_keywords],
                "frequency_counts": dict(top_keywords),
                "total_words": len(words),
                "unique_keywords": len(word_freq)
            }
            
        except Exception as e:
            return {"error": str(e), "result": None}
    
    def _logic_checker_tool(self, statement: str) -> Dict[str, Any]:
        """Check logical statements"""
        
        try:
            # Simple logical statement checking
            statement_lower = statement.lower()
            
            # Check for logical operators
            has_and = "and" in statement_lower
            has_or = "or" in statement_lower
            has_not = "not" in statement_lower
            has_if = "if" in statement_lower
            has_then = "then" in statement_lower
            
            # Basic validity check
            logical_indicators = [has_and, has_or, has_not, has_if, has_then]
            
            analysis = {
                "has_logical_operators": any(logical_indicators),
                "operators_found": {
                    "and": has_and,
                    "or": has_or,
                    "not": has_not,
                    "if": has_if,
                    "then": has_then
                },
                "complexity": "simple" if sum(logical_indicators) <= 1 else "complex",
                "statement": statement
            }
            
            return {"result": analysis}
            
        except Exception as e:
            return {"error": str(e), "result": None}
    
    def _format_checker_tool(self, data: Any, format_type: str) -> Dict[str, Any]:
        """Check data format validity"""
        
        try:
            if format_type == "email":
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                is_valid = bool(re.match(email_pattern, str(data)))
            elif format_type == "url":
                import re
                url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
                is_valid = bool(re.match(url_pattern, str(data)))
            elif format_type == "number":
                try:
                    float(data)
                    is_valid = True
                except ValueError:
                    is_valid = False
            elif format_type == "date":
                import re
                date_pattern = r'^\d{4}-\d{2}-\d{2}$'
                is_valid = bool(re.match(date_pattern, str(data)))
            else:
                return {"error": f"Unknown format type: {format_type}", "result": None}
            
            return {
                "result": is_valid,
                "data": str(data),
                "format_type": format_type
            }
            
        except Exception as e:
            return {"error": str(e), "result": None}
    
    def display_tools(self):
        """Display all available tools"""
        
        console.print("🛠️  NIS Agent Tools", style="bold blue")
        console.print("=" * 50)
        
        for category, tool_names in self.tool_categories.items():
            console.print(f"\n📁 {category.upper()}", style="bold yellow")
            
            for tool_name in tool_names:
                tool = self.tools[tool_name]
                console.print(f"  🔧 {tool_name}: {tool.description}")
                
                # Show parameters
                if tool.parameters:
                    param_str = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
                    console.print(f"     Parameters: {param_str}", style="dim")

# Global tool loader instance
tool_loader = ToolLoader()

def get_tool_loader() -> ToolLoader:
    """Get the global tool loader instance"""
    return tool_loader

def load_agent_tools(agent_dir: Path) -> int:
    """Load tools specific to an agent"""
    tools_dir = agent_dir / "tools"
    return tool_loader.load_tools_from_directory(tools_dir)

def main():
    """Demo of tool loader functionality"""
    
    console.print("🛠️  NIS Tool Loader Demo", style="bold blue")
    
    # Display available tools
    tool_loader.display_tools()
    
    # Test some tools
    console.print("\n🧪 Testing Tools:", style="bold green")
    
    # Test calculator
    calc_result = tool_loader.call_tool("calculator", expression="2 + 3 * 4")
    console.print(f"Calculator: {calc_result}")
    
    # Test text analyzer
    text_result = tool_loader.call_tool("text_analyzer", text="The quick brown fox jumps over the lazy dog")
    console.print(f"Text Analyzer: {text_result}")

if __name__ == "__main__":
    main() 