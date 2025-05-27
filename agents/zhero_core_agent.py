# C:\PROJECTS\ZHERO\ZHEROBE\zhero_adk_backend\agents\zhero_core_agent.py
import logging
from typing import List
from google.adk.agents import LlmAgent
# --- UPDATED: Import Tool from vertexai.generative_models ---
from vertexai.generative_models import Tool 

# Import the instruction from the config file
from config.agent_instructions import DEFAULT_ZHERO_CORE_AGENT_INSTRUCTION

logger = logging.getLogger(__name__)

class ZHeroCoreAgent(LlmAgent):
    def __init__(self, name: str = "ZHeroCoreAgent", model: str = "gemini-1.5-flash", tools: List[Tool] = None):
        super().__init__(
            name=name,
            model=model,
            instruction=DEFAULT_ZHERO_CORE_AGENT_INSTRUCTION, # Use the imported instruction
            tools=tools or [] # Pass the list of Tool instances
        )
        # Fix for logger.info trying to access len(tools) before it's guaranteed to be not None
        num_tools = len(tools) if tools is not None else 0
        logger.info(f"Initialized {self.name} with {num_tools} tools.")