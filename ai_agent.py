import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockAnalysisAgent:
    """
    A LangGraph REACT agent for stock analysis that uses Tavily search tool
    to gather relevant information based on stock data and current market conditions.
    """
    
    def __init__(
        self, 
        openai_api_key: str,
        tavily_api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0
    ):
        """
        Initialize the Stock Analysis Agent.
        
        Args:
            openai_api_key: OpenAI API key for LLM
            tavily_api_key: Tavily API key for search functionality
            model_name: OpenAI model to use
            temperature: Temperature for LLM responses
        """
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature
        )
        
        self.tavily_tool = TavilySearchResults(
            tavily_api_key=tavily_api_key,
            max_results=5
        )
        
        self.tools = [
            Tool(
                name="tavily_search",
                description="Search for current news, industry update, market data, and financial information about stocks and companies",
                func=self.tavily_tool.run
            )
        ]
        
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10  # Increased from 3 to 10
        )
    
    def _create_agent(self) -> Any:
        """Create the REACT agent with custom prompt template."""
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template="""
You are an advanced AI Financial Analyst Agent specializing in comprehensive stock market analysis. Your objective is to provide in-depth, accurate, and actionable insights based on data, news, and financial indicators.

You have access to the following tools:
{tools}

Tool Names: {tool_names}

Your analysis must:
1. Perform a holistic review, integrating:
   - Current market trends and investor sentiment
   - Company-specific news and events
   - Sector and industry performance dynamics
   - Macroeconomic indicators (interest rates, inflation, GDP trends)
   - Technical and fundamental analysis signals
2. Validate findings using objective data from available tools.
3. Provide well-structured, evidence-backed reasoning for every insight. Avoid generic statements.
4. Highlight short-term vs long-term implications where possible.
5. If conflicting signals exist, explain the reasons behind the divergence and give a balanced assessment.
6. Be professional, clear, and concise, avoiding speculation without basis.

**Analysis Framework:**
- Executive Summary: High-level conclusion of the stock’s outlook.
- Key Drivers: Identify and explain the main factors influencing the stock.
- Supporting Evidence: Summarize relevant data, news, and indicators.
- Risk Factors: Highlight potential downside risks or uncertainties.
- Outlook: Provide a reasoned projection (bullish, bearish, neutral) with time horizon context.

**Reasoning and Response Structure:**
Question: The input question you must answer
Thought: Outline your reasoning and next steps
Action: The action to take (must be one of [{tool_names}])
Action Input: The exact input for that action
Observation: The result of the action
...(Repeat Thought/Action/Action Input/Observation as needed)
Thought: Summarize how observations lead to final conclusion
Final Answer: Provide a structured, professional, and comprehensive stock analysis as per the Analysis Framework.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
        )
        
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt_template
        )
    
    def analyze_stocks(
        self, 
        stocks_df: pd.DataFrame, 
        current_date: datetime,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze stocks based on provided dataframe and current date.
        
        Args:
            stocks_df: DataFrame containing stock information
            current_date: Current date for analysis context
            custom_prompt: Optional custom analysis prompt
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Validate input data
            if stocks_df.empty:
                raise ValueError("Stock dataframe cannot be empty")
            
            # Extract key information from dataframe
            stock_symbols = self._extract_stock_symbols(stocks_df)
            stock_summary = self._create_stock_summary(stocks_df)
            
            # Generate analysis prompt
            analysis_prompt = self._generate_analysis_prompt(
                stock_symbols, 
                stock_summary, 
                current_date, 
                custom_prompt
            )
            
            logger.info(f"Analyzing {len(stock_symbols)} stocks: {', '.join(stock_symbols)}")
            
            # Execute agent analysis
            result = self.agent_executor.invoke({"input": analysis_prompt})
            
            return {
                "success": True,
                "analysis": result.get("output", ""),
                "stocks_analyzed": stock_symbols,
                "analysis_date": current_date.isoformat(),
                "stock_count": len(stock_symbols)
            }
            
        except Exception as e:
            logger.error(f"Error in stock analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    def _extract_stock_symbols(self, df: pd.DataFrame) -> List[str]:
        """Extract stock symbols from dataframe."""
        possible_columns = [ 'ticker', 'Ticker']
        
        for col in possible_columns:
            if col in df.columns:
                return df[col].tolist()
        
        # If no symbol column found, use index if it looks like symbols
        if all(isinstance(idx, str) and len(idx) <= 5 for idx in df.index[:5]):
            return df.index.tolist()
        
        raise ValueError("Could not identify stock symbols in dataframe")
    
    def _create_stock_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of stock data for analysis."""
        summary_parts = [
            f"Dataset contains {len(df)} stocks",
            f"Columns available: {', '.join(df.columns)}"
        ]
        
        # Add basic statistics if numerical columns exist
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append(f"Numerical columns: {', '.join(numeric_cols)}")
        
        return ". ".join(summary_parts)
    
    def _generate_analysis_prompt(
        self, 
        symbols: List[str], 
        summary: str, 
        current_date: datetime,
        custom_prompt: Optional[str]
    ) -> str:
        """Generate comprehensive analysis prompt for the agent."""
        
        base_prompt = f"""
        As a financial analyst, analyze the following stocks as of {current_date.strftime('%Y-%m-%d')}:
        
        Stock Symbols: {', '.join(symbols[:10])}  # Limit to first 10 for readability
        Data Summary: {summary}
        
        Please provide analysis on:
        1. Recent market performance and trends for these stocks
        2. Any significant news or events affecting these companies
        3. Sector analysis and industry outlook
        4. Current market sentiment and analyst recommendations
        5. Key risk factors and opportunities
        
        Focus on actionable insights and current market conditions.
        """
        
        if custom_prompt:
            base_prompt += f"\n\nAdditional Analysis Request: {custom_prompt}"
        
        return base_prompt
    
    def ask_custom_question(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a custom question to the agent with optional context.
        
        Args:
            question: The question to ask
            context: Optional context to provide
            
        Returns:
            Dictionary containing the response
        """
        try:
            full_question = question
            if context:
                full_question = f"Context: {context}\n\nQuestion: {question}"
            
            result = self.agent_executor.invoke({"input": full_question})
            
            return {
                "success": True,
                "response": result.get("output", ""),
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error in custom question: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }

